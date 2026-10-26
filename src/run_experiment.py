import argparse
import asyncio
import functools
import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import psutil
import aiohttp

from src import client
from src.client import generate_with_stats
from src.gpu_memory import get_vram_mb
from src.metrics import (
    RequestResult,
    TaskResult,
    save_results_csv,
    summarize_results,
    save_summary_csv,
)
from src.scheduler import FIFOScheduler
from src.workload import make_test_workload


EXPECTED_SERVER_SLOTS = 8


def server_base_url():
    return client.LLM_URL.split("/v1/", 1)[0]


def command_argument(arguments, names):
    for index, argument in enumerate(arguments):
        for name in names:
            if argument == name and index + 1 < len(arguments):
                return arguments[index + 1]
            if argument.startswith(f"{name}="):
                return argument.split("=", 1)[1]
    return None


def safe_server_command(arguments):
    sensitive = {"--api-key", "--api-key-file", "--ssl-key-file"}
    safe = []
    redact_next = False
    for argument in arguments:
        if redact_next:
            safe.append("<redacted>")
            redact_next = False
        elif argument in sensitive:
            safe.append(argument)
            redact_next = True
        elif any(argument.startswith(f"{name}=") for name in sensitive):
            safe.append(argument.split("=", 1)[0] + "=<redacted>")
        else:
            safe.append(argument)
    return safe


async def get_server_slots(session):
    async with session.get(f"{server_base_url()}/slots") as response:
        response.raise_for_status()
        slots = await response.json()

    if not isinstance(slots, list) or len(slots) != EXPECTED_SERVER_SLOTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SERVER_SLOTS} llama-server slots; got "
            f"{len(slots) if isinstance(slots, list) else 'an invalid response'}"
        )
    if any(slot.get("is_processing") for slot in slots):
        raise RuntimeError("llama-server has busy slots; cannot start a clean trial")
    return slots


async def inspect_server():
    process = find_llama_server_process()
    if process is None:
        raise RuntimeError("Start llama-server before running the sweep")

    arguments = process.cmdline()
    if command_argument(arguments, ("-np", "--parallel")) != str(EXPECTED_SERVER_SLOTS):
        raise RuntimeError("Start llama-server with -np 8 for this sweep")
    if command_argument(arguments, ("-cram", "--cache-ram")) != "0":
        raise RuntimeError(
            "Start llama-server with --cache-ram 0 so slot erasure "
            "also clears reusable prompt state"
        )
    if command_argument(arguments, ("--slot-save-path",)) is None:
        raise RuntimeError(
            "Start llama-server with --slot-save-path DIR to enable slot erasure"
        )

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        slots = await get_server_slots(session)

    version = None
    try:
        version_result = subprocess.run(
            [process.exe(), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        version = (version_result.stdout or version_result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired, psutil.Error):
        pass

    return {
        "command": safe_server_command(arguments),
        "version": version,
        "slot_count": len(slots),
        "slot_context_tokens": [slot.get("n_ctx") for slot in slots],
        "cache_reset": "--cache-ram 0 and erase every slot before each trial",
        "endpoint": client.LLM_URL,
    }


async def erase_server_slots():
    erased_tokens = 0
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        slots = await get_server_slots(session)
        for slot in slots:
            slot_id = slot["id"]
            async with session.post(
                f"{server_base_url()}/slots/{slot_id}?action=erase"
            ) as response:
                response.raise_for_status()
                erased_tokens += (await response.json()).get("n_erased", 0)
    return erased_tokens


def find_llama_server_process():
    server_port = str(urlsplit(client.LLM_URL).port or 80)
    for process in psutil.process_iter(
        ["pid", "name"]
    ):
        try:
            name = process.info["name"] or ""

            if not name.startswith("llama-server"):
                continue

            arguments = process.cmdline()
            if (
                command_argument(arguments, ("-np", "--parallel"))
                == str(EXPECTED_SERVER_SLOTS)
                and command_argument(arguments, ("-cram", "--cache-ram"))
                == "0"
                and command_argument(arguments, ("--port",)) == server_port
            ):
                return process

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):
            continue

    return None


def get_process_memory_mb(process):
    if process is None:
        return 0.0

    try:
        memory_bytes = process.memory_info().rss

        return memory_bytes / (1024 * 1024)

    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
    ):
        return 0.0


async def monitor_memory(process, stop_event):
    peak_memory_mb = 0.0
    peak_vram_mb = None
    vram_scope = None

    pid = process.pid if process is not None else None

    while not stop_event.is_set():
        memory_mb = get_process_memory_mb(process)

        peak_memory_mb = max(
            peak_memory_mb,
            memory_mb,
        )

        vram_mb, scope = get_vram_mb(pid)

        if vram_mb is not None:
            peak_vram_mb = max(
                peak_vram_mb or 0.0,
                vram_mb,
            )
            vram_scope = scope

        await asyncio.sleep(0.05)

    return peak_memory_mb, peak_vram_mb, vram_scope


async def call_llm(
    task,
    step,
    prompt,
    max_tokens,
    request_log,
    experiment_start,
):
    start_time = time.perf_counter()

    try:
        content, stats = await generate_with_stats(
            prompt,
            max_tokens=max_tokens,
        )

    except Exception as e:
        request_log.append(
            RequestResult(
                task_id=task.task_id,
                workload_type=task.workload_type,
                step=step,
                start_time=start_time - experiment_start,
                latency=time.perf_counter() - start_time,
                success=False,
                error=f"{type(e).__name__}: {e}",
            )
        )

        raise

    request_log.append(
        RequestResult(
            task_id=task.task_id,
            workload_type=task.workload_type,
            step=step,
            start_time=start_time - experiment_start,
            success=True,
            **stats,
        )
    )

    return content


def request_totals(task, request_log):
    requests = [
        request
        for request in request_log
        if request.task_id == task.task_id
    ]

    return {
        "num_requests": len(requests),
        "output_tokens": sum(
            request.completion_tokens or 0
            for request in requests
        ),
    }


async def run_task(
    task,
    queue_time,
    request_log,
    experiment_start,
):
    start_time = time.perf_counter()

    llama_process = find_llama_server_process()

    stop_event = asyncio.Event()

    memory_monitor = asyncio.create_task(
        monitor_memory(
            llama_process,
            stop_event,
        )
    )

    try:
        if task.workload_type == "multistep":
            first_response = await call_llm(
                task,
                1,
                task.prompt,
                task.max_tokens,
                request_log,
                experiment_start,
            )

            await asyncio.sleep(
                task.tool_delay
            )

            if task.agent_history:
                # Resend the whole conversation: step 2 can reuse
                # step 1's KV only if it is still in a slot.
                second_prompt = [
                    {"role": "user", "content": task.prompt},
                    {"role": "assistant", "content": first_response},
                    {
                        "role": "user",
                        "content": "Now summarize the most important "
                        "point in one sentence.",
                    },
                ]
            else:
                second_prompt = (
                    "Here is an earlier response:\n\n"
                    f"{first_response}\n\n"
                    "Now summarize the most important "
                    "point in one sentence."
                )

            await call_llm(
                task,
                2,
                second_prompt,
                50,
                request_log,
                experiment_start,
            )

        else:
            await call_llm(
                task,
                1,
                task.prompt,
                task.max_tokens,
                request_log,
                experiment_start,
            )

        execution_time = (
            time.perf_counter()
            - start_time
        )

        stop_event.set()

        (
            peak_memory_mb,
            peak_vram_mb,
            vram_scope,
        ) = await memory_monitor

        return TaskResult(
            task_id=task.task_id,
            workload_type=task.workload_type,
            queue_time=queue_time,
            execution_time=execution_time,
            completion_time=(
                queue_time + execution_time
            ),
            peak_memory_mb=peak_memory_mb,
            success=True,
            peak_vram_mb=peak_vram_mb,
            vram_scope=vram_scope,
            **request_totals(task, request_log),
        )

    except Exception as e:
        execution_time = (
            time.perf_counter()
            - start_time
        )

        stop_event.set()

        (
            peak_memory_mb,
            peak_vram_mb,
            vram_scope,
        ) = await memory_monitor

        return TaskResult(
            task_id=task.task_id,
            workload_type=task.workload_type,
            queue_time=queue_time,
            execution_time=execution_time,
            completion_time=(
                queue_time + execution_time
            ),
            peak_memory_mb=peak_memory_mb,
            success=False,
            error=f"{type(e).__name__}: {e}",
            peak_vram_mb=peak_vram_mb,
            vram_scope=vram_scope,
            timed_out=isinstance(
                e,
                asyncio.TimeoutError,
            ),
            **request_totals(task, request_log),
        )


async def run_experiment(
    concurrency,
    trial,
    tasks_per_type,
    context_repeats,
    output_dir,
    tool_delay=2.0,
    agent_history=False,
):
    tasks = make_test_workload(
        tasks_per_type=tasks_per_type,
        context_repeats=context_repeats,
        tool_delay=tool_delay,
        agent_history=agent_history,
    )

    scheduler = FIFOScheduler(
        concurrency=concurrency
    )

    print(
        f"\nRunning trial {trial} "
        f"with concurrency={concurrency}, "
        f"context_repeats={context_repeats}\n"
    )

    request_log = []

    experiment_start = time.perf_counter()

    results = await scheduler.run(
        tasks,
        functools.partial(
            run_task,
            request_log=request_log,
            experiment_start=experiment_start,
        ),
    )

    total_time = (
        time.perf_counter()
        - experiment_start
    )

    for result in results:
        print(
            f"{result.task_id}: "
            f"queue={result.queue_time:.3f}s, "
            f"exec={result.execution_time:.3f}s, "
            f"total={result.completion_time:.3f}s, "
            f"memory={result.peak_memory_mb:.1f} MB, "
            f"vram={result.peak_vram_mb or 0:.1f} MB, "
            f"success={result.success}"
            + (
                f" ({result.error})"
                if result.error else ""
            )
        )

    print(
        f"\nTotal experiment time: "
        f"{total_time:.3f}s"
    )

    output_prefix = output_dir / (
        f"ctx_{context_repeats}_"
        f"concurrency_{concurrency}_"
        f"trial_{trial}"
    )

    save_results_csv(
        results,
        f"{output_prefix}.csv",
    )

    save_results_csv(
        sorted(
            request_log,
            key=lambda request: request.start_time,
        ),
        f"{output_prefix}_requests.csv",
    )

    print(
        f"Saved results to {output_prefix}.csv "
        f"and {output_prefix}_requests.csv"
    )

    summary = summarize_results(
        results,
        concurrency,
        total_time,
        requests=request_log,
    )

    summary["trial"] = trial
    summary["context_repeats"] = context_repeats

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the FIFO baseline across "
            "concurrency levels and context sizes."
        )
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the reproducible trial order.",
    )

    parser.add_argument(
        "--tasks-per-type",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--context-repeats",
        type=int,
        nargs="+",
        default=[30, 60, 120],
        help=(
            "Sizes of the long_context prompt "
            "(1k/2k/4k target labels; actual token counts are recorded)."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory for this sweep (default: unique directory under results/runs).",
    )

    parser.add_argument(
        "--server-url",
        default=client.LLM_URL,
        help="llama-server chat completions URL.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=client.REQUEST_TIMEOUT,
        help="Per-request timeout in seconds.",
    )

    parser.add_argument(
        "--tool-delay",
        type=float,
        default=2.0,
        help="Seconds a multistep task waits between its two LLM calls.",
    )

    parser.add_argument(
        "--agent-history",
        action="store_true",
        help=(
            "Multistep tasks carry the long context and resend the whole "
            "conversation in step 2, so step 2 can reuse step 1's KV."
        ),
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    client.LLM_URL = args.server_url
    client.REQUEST_TIMEOUT = args.timeout
    server_info = await inspect_server()

    output_dir = args.output_dir or (
        Path("results") / "runs" / (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid4().hex[:8]
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Writing this sweep to {output_dir}")
    conditions = [
        (context_repeats, concurrency, trial)
        for context_repeats in args.context_repeats
        for concurrency in args.concurrency
        for trial in range(1, args.trials + 1)
    ]
    random.Random(args.seed).shuffle(conditions)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "started_at": datetime.now().astimezone().isoformat(),
                "concurrency": args.concurrency,
                "context_repeats": args.context_repeats,
                "trials": args.trials,
                "seed": args.seed,
                "condition_order": [
                    {
                        "context_repeats": context_repeats,
                        "concurrency": concurrency,
                        "trial": trial,
                    }
                    for context_repeats, concurrency, trial in conditions
                ],
                "tasks_per_type": args.tasks_per_type,
                "tool_delay_seconds": args.tool_delay,
                "agent_history": args.agent_history,
                "request_timeout_seconds": args.timeout,
                "server": server_info,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summaries = []

    for context_repeats, concurrency, trial in conditions:
        erased_tokens = await erase_server_slots()
        print(f"Cleared {erased_tokens} slot tokens before trial {trial}")
        summary = await run_experiment(
            concurrency,
            trial,
            args.tasks_per_type,
            context_repeats,
            output_dir,
            tool_delay=args.tool_delay,
            agent_history=args.agent_history,
        )

        summary["tool_delay"] = args.tool_delay
        summary["agent_history"] = args.agent_history
        summary["slot_tokens_erased_before_trial"] = erased_tokens
        summaries.append(summary)
        save_summary_csv(
            summaries,
            output_dir / "summary_trials.csv",
        )

    print(
        "\nSaved all trial summaries "
        f"to {output_dir / 'summary_trials.csv'}"
    )
    print(
        "Analyze with: python analysis/analyze.py "
        f'--results-dir "{output_dir}"'
    )


if __name__ == "__main__":
    asyncio.run(main())
