import glob
import os

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CONTEXT_REPEATS = 120


def load_trials():
    df = pd.read_csv(
        "results/summary_trials.csv"
    )

    # Older runs did not vary context size.
    if "context_repeats" not in df.columns:
        df["context_repeats"] = (
            DEFAULT_CONTEXT_REPEATS
        )

    return df


def load_requests():
    paths = glob.glob(
        "results/ctx_*_concurrency_*_trial_*_requests.csv"
    )

    if not paths:
        return None

    frames = []

    for path in paths:
        name = os.path.basename(path).split("_")

        frame = pd.read_csv(path)
        frame["context_repeats"] = int(name[1])
        frame["concurrency"] = int(name[3])
        frame["trial"] = int(name[5])

        frames.append(frame)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def aggregate_trials(df):
    aggregations = {
        "mean_throughput": ("throughput", "mean"),
        "std_throughput": ("throughput", "std"),
        "mean_completion_time": ("mean_completion_time", "mean"),
        "std_completion_time": ("mean_completion_time", "std"),
        "mean_queue_time": ("mean_queue_time", "mean"),
        "mean_execution_time": ("mean_execution_time", "mean"),
        "mean_peak_memory_mb": ("mean_peak_memory_mb", "mean"),
        "max_peak_memory_mb": ("max_peak_memory_mb", "max"),
    }

    optional_aggregations = {
        "p50_completion_time": ("p50_completion_time", "mean"),
        "p95_completion_time": ("p95_completion_time", "mean"),
        "p99_completion_time": ("p99_completion_time", "mean"),
        "mean_peak_vram_mb": ("mean_peak_vram_mb", "mean"),
        "max_peak_vram_mb": ("max_peak_vram_mb", "max"),
        "total_tasks": ("num_tasks", "sum"),
        "failed_tasks": ("failed_tasks", "sum"),
        "timed_out_tasks": ("timed_out_tasks", "sum"),
        "mean_output_tokens_per_trial": ("total_output_tokens", "mean"),
        "mean_long_context_prompt_tokens": (
            "mean_long_context_prompt_tokens",
            "mean",
        ),
    }

    for name, (column, func) in optional_aggregations.items():
        if column in df.columns:
            aggregations[name] = (column, func)

    aggregated = (
        df.groupby(["context_repeats", "concurrency"])
        .agg(**aggregations)
        .reset_index()
    )

    if "failed_tasks" in aggregated.columns:
        aggregated["failure_rate"] = (
            aggregated["failed_tasks"]
            / aggregated["total_tasks"]
        )

    return aggregated


def aggregate_requests(requests):
    successful = requests[requests["success"]]

    return (
        successful.groupby(
            [
                "context_repeats",
                "concurrency",
                "workload_type",
                "step",
            ]
        )
        .agg(
            num_requests=("latency", "count"),
            mean_latency=("latency", "mean"),
            p95_latency=(
                "latency",
                lambda values: values.quantile(0.95),
            ),
            mean_prefill_ms=("prefill_ms", "mean"),
            mean_decode_ms=("decode_ms", "mean"),
            mean_prompt_tokens=("prompt_tokens", "mean"),
            mean_cached_tokens=("cached_tokens", "mean"),
            mean_completion_tokens=("completion_tokens", "mean"),
        )
        .reset_index()
    )


def context_label(df, context_repeats):
    if (
        "mean_long_context_prompt_tokens" in df.columns
    ):
        tokens = df.loc[
            df["context_repeats"] == context_repeats,
            "mean_long_context_prompt_tokens",
        ].mean()

        if pd.notna(tokens):
            return f"long_context ≈ {tokens:.0f} tokens"

    return f"context_repeats = {context_repeats}"


def plot_per_context(
    df,
    column,
    ylabel,
    title,
    filename,
    error_column=None,
):
    if column not in df.columns or df[column].isna().all():
        return

    plt.figure()

    for context_repeats, group in df.groupby(
        "context_repeats"
    ):
        label = context_label(df, context_repeats)

        if error_column:
            plt.errorbar(
                group["concurrency"],
                group[column],
                yerr=group[error_column],
                marker="o",
                capsize=4,
                label=label,
            )

        else:
            plt.plot(
                group["concurrency"],
                group[column],
                marker="o",
                label=label,
            )

    plt.xlabel("Concurrency")
    plt.ylabel(ylabel)
    plt.title(title)

    if df["context_repeats"].nunique() > 1:
        plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"results/{filename}",
        dpi=200,
    )

    plt.close()


def plot_memory(df, prefix, label, filename):
    mean_column = f"mean_peak_{prefix}_mb"
    max_column = f"max_peak_{prefix}_mb"

    if (
        mean_column not in df.columns
        or df[mean_column].isna().all()
    ):
        return

    plt.figure()

    multiple = df["context_repeats"].nunique() > 1

    for context_repeats, group in df.groupby(
        "context_repeats"
    ):
        suffix = (
            f" ({context_label(df, context_repeats)})"
            if multiple else ""
        )

        plt.plot(
            group["concurrency"],
            group[mean_column],
            marker="o",
            label=f"Mean peak {label}{suffix}",
        )

        plt.plot(
            group["concurrency"],
            group[max_column],
            marker="o",
            linestyle="--",
            label=f"Max peak {label}{suffix}",
        )

    plt.xlabel("Concurrency")
    plt.ylabel(f"{label} Usage (MB)")
    plt.title(f"{label} Usage vs Concurrency")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"results/{filename}",
        dpi=200,
    )

    plt.close()


def plot_tail_latency(df):
    if "p95_completion_time" not in df.columns:
        return

    plt.figure()

    multiple = df["context_repeats"].nunique() > 1

    for context_repeats, group in df.groupby(
        "context_repeats"
    ):
        suffix = (
            f" ({context_label(df, context_repeats)})"
            if multiple else ""
        )

        for column, label, style in [
            ("p50_completion_time", "p50", "-"),
            ("p95_completion_time", "p95", "--"),
            ("p99_completion_time", "p99", ":"),
        ]:
            plt.plot(
                group["concurrency"],
                group[column],
                marker="o",
                linestyle=style,
                label=f"{label}{suffix}",
            )

    plt.xlabel("Concurrency")
    plt.ylabel("Completion Time (seconds)")
    plt.title(
        "Completion Time Percentiles vs Concurrency"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/"
        "tail_latency_vs_concurrency.png",
        dpi=200,
    )

    plt.close()


def main():
    trials = load_trials()

    aggregated = aggregate_trials(
        trials
    )

    aggregated.to_csv(
        "results/"
        "summary_aggregated.csv",
        index=False,
    )

    print(
        "\nAggregated results:"
    )

    print(
        aggregated.to_string(
            index=False
        )
    )

    plot_per_context(
        aggregated,
        "mean_throughput",
        "Throughput (tasks/second)",
        "Throughput vs Concurrency",
        "throughput_vs_concurrency.png",
        error_column="std_throughput",
    )

    plot_per_context(
        aggregated,
        "mean_completion_time",
        "Mean Completion Time (seconds)",
        "Completion Time vs Concurrency",
        "completion_time_vs_concurrency.png",
        error_column="std_completion_time",
    )

    plot_per_context(
        aggregated,
        "mean_queue_time",
        "Mean Queue Time (seconds)",
        "Queue Time vs Concurrency",
        "queue_time_vs_concurrency.png",
    )

    plot_per_context(
        aggregated,
        "mean_execution_time",
        "Mean Execution Time (seconds)",
        "Execution Time vs Concurrency",
        "execution_time_vs_concurrency.png",
    )

    plot_per_context(
        aggregated,
        "failure_rate",
        "Failure Rate",
        "Failure Rate vs Concurrency",
        "failure_rate_vs_concurrency.png",
    )

    plot_tail_latency(
        aggregated
    )

    plot_memory(
        aggregated,
        "memory",
        "Memory",
        "memory_vs_concurrency.png",
    )

    plot_memory(
        aggregated,
        "vram",
        "VRAM",
        "vram_vs_concurrency.png",
    )

    print(
        "\nSaved aggregated results "
        "to results/"
        "summary_aggregated.csv"
    )

    requests = load_requests()

    if requests is not None:
        request_summary = aggregate_requests(
            requests
        )

        request_summary.to_csv(
            "results/"
            "request_summary.csv",
            index=False,
        )

        print(
            "\nPer-request results:"
        )

        print(
            request_summary.to_string(
                index=False
            )
        )

        print(
            "\nSaved per-request results "
            "to results/request_summary.csv"
        )

    print(
        "Saved updated plots "
        "with repeated-trial results."
    )


if __name__ == "__main__":
    main()
