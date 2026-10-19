import asyncio
import time

import psutil

from src.client import generate
from src.metrics import (
    TaskResult,
    save_results_csv,
    summarize_results,
    save_summary_csv,
)
from src.scheduler import FIFOScheduler
from src.workload import make_test_workload


def find_llama_server_process():
    for process in psutil.process_iter(
        ["pid", "name", "cmdline"]
    ):
        try:
            name = process.info["name"] or ""
            cmdline = process.info["cmdline"] or []

            command = " ".join(cmdline)

            if (
                "llama-server" in name
                or "llama-server" in command
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

    while not stop_event.is_set():
        memory_mb = get_process_memory_mb(process)

        peak_memory_mb = max(
            peak_memory_mb,
            memory_mb,
        )

        await asyncio.sleep(0.05)

    return peak_memory_mb


async def run_task(task, queue_time):
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
            first_response = await generate(
                task.prompt,
                max_tokens=task.max_tokens,
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

            await generate(
                second_prompt,
                max_tokens=50,
            )

        else:
            await generate(
                task.prompt,
                max_tokens=task.max_tokens,
            )

        execution_time = (
            time.perf_counter()
            - start_time
        )

        stop_event.set()

        peak_memory_mb = await memory_monitor

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
        )

    except Exception as e:
        execution_time = (
            time.perf_counter()
            - start_time
        )

        stop_event.set()

        peak_memory_mb = await memory_monitor

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
            error=str(e),
        )


async def run_experiment(concurrency, trial):
    tasks = make_test_workload()

    scheduler = FIFOScheduler(
        concurrency=concurrency
    )

    print(
        f"\nRunning trial {trial} "
        f"with concurrency={concurrency}\n"
    )

    experiment_start = time.perf_counter()

    results = await scheduler.run(
        tasks,
        run_task,
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
            f"success={result.success}"
        )

    print(
        f"\nTotal experiment time: "
        f"{total_time:.3f}s"
    )

    output_path = (
        f"results/"
        f"concurrency_{concurrency}_"
        f"trial_{trial}.csv"
    )

    save_results_csv(
        results,
        output_path,
    )

    print(
        f"Saved results to {output_path}"
    )

    summary = summarize_results(
        results,
        concurrency,
        total_time,
    )

    summary["trial"] = trial

    return summary


async def main():
    concurrency_levels = [
        1,
        2,
        4,
        8,
    ]

    num_trials = 3

    summaries = []

    for concurrency in concurrency_levels:
        for trial in range(1, num_trials + 1):
            summary = await run_experiment(
                concurrency,
                trial,
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