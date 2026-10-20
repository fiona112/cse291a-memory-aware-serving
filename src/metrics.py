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
    peak_vram_mb: Optional[float] = None
    vram_scope: Optional[str] = None
    timed_out: bool = False
    num_requests: int = 0
    output_tokens: int = 0


@dataclass
class RequestResult:
    task_id: str
    workload_type: str
    step: int
    start_time: float
    latency: float
    success: bool
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    prefill_ms: Optional[float] = None
    decode_ms: Optional[float] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None


def percentile(values, p):
    """
    Linear-interpolated percentile (p in 0-100),
    same as numpy's default method.
    """
    if not values:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower

    return (
        ordered[lower] * (1 - weight)
        + ordered[upper] * weight
    )


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
    requests=None,
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

    peak_vrams = [
        result.peak_vram_mb
        for result in results
        if result.peak_vram_mb is not None
    ]

    timed_out = sum(
        result.timed_out
        for result in results
    )

    request_latencies = [
        request.latency
        for request in requests or []
        if request.success
    ]

    long_context_prompt_tokens = [
        request.prompt_tokens
        for request in requests or []
        if request.workload_type == "long_context"
        and request.prompt_tokens is not None
    ]

    return {
        "concurrency": concurrency,
        "num_tasks": len(results),
        "successful_tasks": successful,
        "failed_tasks": len(results) - successful,
        "timed_out_tasks": timed_out,
        "failure_rate": (
            (len(results) - successful)
            / len(results)
        ),
        "total_time": total_time,
        "throughput": (
            successful / total_time
        ),
        "mean_completion_time":
            statistics.mean(
                completion_times
            ),
        "p50_completion_time":
            percentile(
                completion_times, 50
            ),
        "p95_completion_time":
            percentile(
                completion_times, 95
            ),
        "p99_completion_time":
            percentile(
                completion_times, 99
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
        "mean_peak_vram_mb":
            statistics.mean(
                peak_vrams
            ) if peak_vrams else None,
        "max_peak_vram_mb":
            max(
                peak_vrams
            ) if peak_vrams else None,
        "vram_scope":
            results[0].vram_scope,
        "num_requests": len(requests or []),
        "mean_request_latency":
            statistics.mean(
                request_latencies
            ) if request_latencies else None,
        "p95_request_latency":
            percentile(
                request_latencies, 95
            ),
        "p99_request_latency":
            percentile(
                request_latencies, 99
            ),
        "total_output_tokens": sum(
            result.output_tokens
            for result in results
        ),
        "mean_long_context_prompt_tokens":
            statistics.mean(
                long_context_prompt_tokens
            ) if long_context_prompt_tokens else None,
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