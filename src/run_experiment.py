import argparse
import asyncio
import functools
import time

import psutil

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


def find_llama_server_process():
    for process in psutil.process_iter(
        ["pid", "name"]
    ):
        try:
            name = process.info["name"] or ""

            # Match on the executable name only: shells or
            # scripts that launched the server also have
            # "llama-server" in their command line.
            if name.startswith("llama-server"):
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
):
    tasks = make_test_workload(
        tasks_per_type=tasks_per_type,
        context_repeats=context_repeats,
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

    output_prefix = (
        f"results/"
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
        default=[1, 2, 4, 8],
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=3,
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
        default=[120],
        help=(
            "Sizes of the long_context prompt "
            "(each repeat is ~32 tokens)."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=client.REQUEST_TIMEOUT,
        help="Per-request timeout in seconds.",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    client.REQUEST_TIMEOUT = args.timeout

    summaries = []

    for context_repeats in args.context_repeats:
        for concurrency in args.concurrency:
            for trial in range(1, args.trials + 1):
                summary = await run_experiment(
                    concurrency,
                    trial,
                    args.tasks_per_type,
                    context_repeats,
                )

                summaries.append(summary)

    save_summary_csv(
        summaries,
        "results/summary_trials.csv",
    )

    print(
        "\nSaved all trial summaries "
        "to results/summary_trials.csv"
    )


if __name__ == "__main__":
    asyncio.run(main())