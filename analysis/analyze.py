import argparse
import glob
import os
from html import escape
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CONTEXT_REPEATS = 120
CONTEXT_TARGET_LABELS = {30: "1k", 60: "2k", 90: "3k", 120: "4k"}


def load_trials(results_dir):
    df = pd.read_csv(
        results_dir / "summary_trials.csv"
    )

    # Older runs did not vary context size.
    if "context_repeats" not in df.columns:
        df["context_repeats"] = (
            DEFAULT_CONTEXT_REPEATS
        )

    return df


def load_requests(trials, results_dir):
    current_trials = set(
        zip(
            trials["context_repeats"],
            trials["concurrency"],
            trials["trial"],
        )
    )

    paths = glob.glob(
        str(results_dir / "ctx_*_concurrency_*_trial_*_requests.csv")
    )

    if not paths:
        return None

    frames = []

    for path in paths:
        name = os.path.basename(path).split("_")

        context_repeats = int(name[1])
        concurrency = int(name[3])
        trial = int(name[5])

        if (context_repeats, concurrency, trial) not in current_trials:
            continue

        frame = pd.read_csv(path)
        frame["context_repeats"] = context_repeats
        frame["concurrency"] = concurrency
        frame["trial"] = trial

        frames.append(frame)

    if not frames:
        return None

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
    if context_repeats in CONTEXT_TARGET_LABELS:
        return CONTEXT_TARGET_LABELS[context_repeats]

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


def format_concurrency_axis(df):
    levels = sorted(df["concurrency"].unique())
    plt.xscale("log", base=2)
    plt.xticks(levels, [str(level) for level in levels])
    plt.xlabel("Concurrency (log₂ scale)")


def plot_per_context(
    df,
    column,
    ylabel,
    title,
    filename,
    results_dir,
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

    format_concurrency_axis(df)
    plt.ylabel(ylabel)
    plt.title(title)

    if df["context_repeats"].nunique() > 1:
        plt.legend()

    plt.tight_layout()

    plt.savefig(
        results_dir / filename,
        dpi=200,
    )

    plt.close()


def plot_memory(df, prefix, label, filename, results_dir):
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

    format_concurrency_axis(df)
    plt.ylabel(f"{label} Usage (MB)")
    plt.title(f"{label} Usage vs Concurrency")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / filename,
        dpi=200,
    )

    plt.close()


def plot_tail_latency(df, results_dir):
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

    format_concurrency_axis(df)
    plt.ylabel("Completion Time (seconds)")
    plt.title(
        "Completion Time Percentiles vs Concurrency"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "tail_latency_vs_concurrency.png",
        dpi=200,
    )

    plt.close()


def write_report(results_dir, trials, aggregated, request_summary):
    overview_columns = [
        "context_repeats",
        "concurrency",
        "mean_long_context_prompt_tokens",
        "mean_throughput",
        "p50_completion_time",
        "p95_completion_time",
        "failure_rate",
        "mean_peak_memory_mb",
        "mean_peak_vram_mb",
    ]
    overview = aggregated[
        [column for column in overview_columns if column in aggregated]
    ].copy()
    overview.insert(
        0,
        "context_target",
        overview["context_repeats"].map(CONTEXT_TARGET_LABELS),
    )
    best_observed = aggregated.loc[
        aggregated.groupby("context_repeats")["mean_throughput"].idxmax(),
        ["context_repeats", "concurrency", "mean_throughput"],
    ].copy()
    best_observed.insert(
        0,
        "context_target",
        best_observed["context_repeats"].map(CONTEXT_TARGET_LABELS),
    )
    chart_files = [
        "throughput_vs_concurrency.png",
        "tail_latency_vs_concurrency.png",
        "completion_time_vs_concurrency.png",
        "queue_time_vs_concurrency.png",
        "execution_time_vs_concurrency.png",
        "failure_rate_vs_concurrency.png",
        "memory_vs_concurrency.png",
        "vram_vs_concurrency.png",
    ]
    charts = "\n".join(
        f'<figure><a href="{name}"><img src="{name}" alt="{escape(name)}" loading="lazy"></a>'
        f'<figcaption>{escape(name.replace("_", " ").removesuffix(".png"))}</figcaption></figure>'
        for name in chart_files
        if (results_dir / name).exists()
    )
    request_table = (
        request_summary.to_html(index=False, float_format=lambda x: f"{x:.3f}")
        if request_summary is not None
        else "<p>No matching per-request CSVs were found for these trials.</p>"
    )
    request_link = (
        '<a href="request_summary.csv">Request summary CSV</a>'
        if request_summary is not None
        else ""
    )
    config_link = (
        '<a href="run_config.json">Run configuration</a>'
        if (results_dir / "run_config.json").exists()
        else ""
    )
    total_tasks = (
        int(trials["num_tasks"].sum())
        if "num_tasks" in trials
        else "unknown"
    )
    title = escape(results_dir.name)
    targets = ", ".join(
        CONTEXT_TARGET_LABELS.get(int(repeats), f"{repeats} repeats")
        for repeats in sorted(trials["context_repeats"].unique())
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment report · {title}</title>
<style>
body {{ margin: 0; background: #f4f6f8; color: #17212b; font: 15px/1.5 system-ui, sans-serif; }}
main {{ max-width: 1400px; margin: auto; padding: 28px; }}
h1 {{ margin-bottom: 0; }}
.muted {{ color: #52606d; }}
.links {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 18px 0 28px; }}
a {{ color: #075da8; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 18px; }}
figure {{ margin: 0; padding: 12px; background: white; border: 1px solid #dce3e9; border-radius: 10px; }}
figure img {{ display: block; width: 100%; height: auto; }}
figcaption {{ padding: 8px 4px 2px; text-transform: capitalize; color: #52606d; }}
.table {{ overflow-x: auto; background: white; border: 1px solid #dce3e9; border-radius: 10px; }}
table {{ border-collapse: collapse; width: 100%; white-space: nowrap; }}
th, td {{ padding: 7px 10px; border-bottom: 1px solid #e8edf1; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #eaf0f5; }}
section {{ margin-top: 34px; }}
details {{ margin-top: 20px; }}
@media (max-width: 500px) {{ main {{ padding: 14px; }} .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body><main>
<h1>Experiment report</h1>
<p class="muted">{title} · {len(trials)} trials · {total_tasks} tasks ·
context targets {escape(targets)} ·
concurrency {escape(', '.join(map(str, sorted(trials['concurrency'].unique()))))}</p>
<nav class="links">
<a href="summary_trials.csv">Trial CSV</a>
<a href="summary_aggregated.csv">Aggregated CSV</a>
{request_link}
{config_link}
</nav>
<section><h2>Overview</h2><div class="table">{overview.to_html(index=False, float_format=lambda x: f'{x:.3f}')}</div></section>
<section><h2>Highest observed throughput</h2><p class="muted">Best among the tested concurrency settings for each context size. Check latency and failures before choosing a baseline.</p><div class="table">{best_observed.to_html(index=False, float_format=lambda x: f'{x:.3f}')}</div></section>
<section><h2>Charts</h2><div class="grid">{charts}</div></section>
<section><h2>Per-request summary</h2><div class="table">{request_table}</div></section>
<details><summary>All trial rows</summary><div class="table">{trials.to_html(index=False, float_format=lambda x: f'{x:.3f}')}</div></details>
</main></body></html>"""
    (results_dir / "report.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze one experiment sweep."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing summary_trials.csv (default: results).",
    )
    results_dir = parser.parse_args().results_dir
    trials = load_trials(results_dir)

    aggregated = aggregate_trials(
        trials
    )

    aggregated.to_csv(
        results_dir / "summary_aggregated.csv",
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
        results_dir,
        error_column="std_throughput",
    )

    plot_per_context(
        aggregated,
        "mean_completion_time",
        "Mean Completion Time (seconds)",
        "Completion Time vs Concurrency",
        "completion_time_vs_concurrency.png",
        results_dir,
        error_column="std_completion_time",
    )

    plot_per_context(
        aggregated,
        "mean_queue_time",
        "Mean Queue Time (seconds)",
        "Queue Time vs Concurrency",
        "queue_time_vs_concurrency.png",
        results_dir,
    )

    plot_per_context(
        aggregated,
        "mean_execution_time",
        "Mean Execution Time (seconds)",
        "Execution Time vs Concurrency",
        "execution_time_vs_concurrency.png",
        results_dir,
    )

    plot_per_context(
        aggregated,
        "failure_rate",
        "Failure Rate",
        "Failure Rate vs Concurrency",
        "failure_rate_vs_concurrency.png",
        results_dir,
    )

    plot_tail_latency(aggregated, results_dir)

    plot_memory(
        aggregated,
        "memory",
        "Memory",
        "memory_vs_concurrency.png",
        results_dir,
    )

    plot_memory(
        aggregated,
        "vram",
        "VRAM",
        "vram_vs_concurrency.png",
        results_dir,
    )

    print(
        "\nSaved aggregated results "
        f"to {results_dir / 'summary_aggregated.csv'}"
    )

    requests = load_requests(trials, results_dir)
    request_summary = None

    if requests is not None:
        request_summary = aggregate_requests(
            requests
        )

        request_summary.to_csv(
            results_dir / "request_summary.csv",
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
            f"to {results_dir / 'request_summary.csv'}"
        )

    write_report(results_dir, trials, aggregated, request_summary)

    print(
        "Saved updated plots "
        f"and report to {results_dir}."
    )


if __name__ == "__main__":
    main()
