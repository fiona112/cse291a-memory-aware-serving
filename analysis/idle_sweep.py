"""
Compare runs that differ only in the multistep tool delay.

For each tool delay and concurrency level it reports throughput, task
latency, and, for multistep step 2 (run with --agent-history), how much
of step 1's KV was still in a slot when the tool returned:
  reuse        = cached tokens / prompt tokens
  recomputed   = prompt tokens - cached tokens
  full_hits    = share of step-2 requests that reused >= 90% of the prompt

Usage:
  python analysis/idle_sweep.py results/runs/idle_sweep_delay0.5 \
      results/runs/idle_sweep_delay2 results/runs/idle_sweep_delay8
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def load_run(results_dir):
    with open(os.path.join(results_dir, "run_config.json")) as f:
        tool_delay = json.load(f)["tool_delay_seconds"]

    trials = pd.read_csv(os.path.join(results_dir, "summary_trials.csv"))
    trials["tool_delay"] = tool_delay

    frames = []
    for path in glob.glob(
        os.path.join(results_dir, "ctx_*_concurrency_*_trial_*_requests.csv")
    ):
        parts = os.path.basename(path).split("_")
        frame = pd.read_csv(path)
        frame["concurrency"] = int(parts[3])
        frames.append(frame)

    requests = pd.concat(frames, ignore_index=True)
    requests["tool_delay"] = tool_delay

    return trials, requests


def summarize(trials, requests):
    task_summary = (
        trials.groupby(["tool_delay", "concurrency"])
        .agg(
            throughput=("throughput", "mean"),
            throughput_std=("throughput", "std"),
            p50_completion_s=("p50_completion_time", "mean"),
            p95_completion_s=("p95_completion_time", "mean"),
            failed_tasks=("failed_tasks", "sum"),
        )
        .reset_index()
    )

    step2 = requests[
        (requests["workload_type"] == "multistep")
        & (requests["step"] == 2)
        & requests["success"]
    ].copy()
    step2["reuse"] = step2["cached_tokens"] / step2["prompt_tokens"]
    step2["recomputed"] = step2["prompt_tokens"] - step2["cached_tokens"]

    reuse_summary = (
        step2.groupby(["tool_delay", "concurrency"])
        .agg(
            step2_reuse=("reuse", "mean"),
            step2_full_hits=("reuse", lambda s: (s >= 0.9).mean()),
            step2_recomputed_tokens=("recomputed", "mean"),
            step2_prefill_ms=("prefill_ms", "mean"),
        )
        .reset_index()
    )

    return task_summary.merge(
        reuse_summary,
        on=["tool_delay", "concurrency"],
        how="left",
    )


def plot(summary, column, ylabel, title, path):
    plt.figure()

    for tool_delay, group in summary.groupby("tool_delay"):
        plt.plot(
            group["concurrency"],
            group[column],
            marker="o",
            label=f"tool delay {tool_delay:g} s",
        )

    plt.xscale("log", base=2)
    plt.xticks(
        sorted(summary["concurrency"].unique()),
        [str(c) for c in sorted(summary["concurrency"].unique())],
    )
    plt.xlabel("Concurrency")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("results_dirs", nargs="+")
    parser.add_argument("--out", default="results/idle_sweep")
    args = parser.parse_args()

    loaded = [load_run(d) for d in args.results_dirs]
    trials = pd.concat([t for t, _ in loaded], ignore_index=True)
    requests = pd.concat([r for _, r in loaded], ignore_index=True)

    summary = summarize(trials, requests).sort_values(
        ["tool_delay", "concurrency"]
    )

    os.makedirs(args.out, exist_ok=True)
    summary.to_csv(os.path.join(args.out, "idle_sweep_summary.csv"), index=False)

    plot(
        summary,
        "throughput",
        "Throughput (tasks/second)",
        "Throughput vs Concurrency by Tool Delay",
        os.path.join(args.out, "throughput_by_tool_delay.png"),
    )
    plot(
        summary,
        "step2_reuse",
        "Step-2 KV reuse (cached / prompt tokens)",
        "KV Retained Across the Tool Call",
        os.path.join(args.out, "step2_reuse_by_tool_delay.png"),
    )

    print(summary.round(3).to_string(index=False))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
