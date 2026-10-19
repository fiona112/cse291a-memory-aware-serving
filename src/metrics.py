from dataclasses import dataclass, asdict
from typing import Optional
import csv
import statistics


@dataclass
class TaskResult:
    task_id: str
    workload_type: str
    queue_time: float
    execution_time: float
    completion_time: float
    peak_memory_mb: float
    success: bool
    error: Optional[str] = None


def save_results_csv(results, output_path):
    if not results:
        return

    fieldnames = list(
        asdict(results[0]).keys()
    )

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                asdict(result)
            )


def summarize_results(
    results,
    concurrency,
    total_time,
):
    completion_times = [
        result.completion_time
        for result in results
    ]

    queue_times = [
        result.queue_time
        for result in results
    ]

    execution_times = [
        result.execution_time
        for result in results
    ]

    peak_memories = [
        result.peak_memory_mb
        for result in results
    ]

    successful = sum(
        result.success
        for result in results
    )

    return {
        "concurrency": concurrency,
        "num_tasks": len(results),
        "successful_tasks": successful,
        "total_time": total_time,
        "throughput": (
            successful / total_time
        ),
        "mean_completion_time":
            statistics.mean(
                completion_times
            ),
        "mean_queue_time":
            statistics.mean(
                queue_times
            ),
        "mean_execution_time":
            statistics.mean(
                execution_times
            ),
        "max_completion_time":
            max(
                completion_times
            ),
        "mean_peak_memory_mb":
            statistics.mean(
                peak_memories
            ),
        "max_peak_memory_mb":
            max(
                peak_memories
            ),
    }


def save_summary_csv(
    summaries,
    output_path,
):
    if not summaries:
        return

    fieldnames = list(
        summaries[0].keys()
    )

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )