"""
Compare KV policies from src.agent_replay runs.

Reads every <results-root>/*/summary_trials.csv and the per-episode
CSVs, aggregates over trials, and writes:
  policy_summary.csv    mean (and std) of each metric per policy and rate
  fairness.csv          non-tool time of the longest third of episodes
  p95_completion_by_rate.png, computed_tokens_by_rate.png

Usage:
  python analysis/compare_policies.py results/replay/m3
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

METRICS = [
    "makespan_s",
    "throughput_eps",
    "p50_completion_s",
    "p95_completion_s",
    "p99_completion_s",
    "mean_non_tool_s",
    "p95_non_tool_s",
    "computed_tokens",
    "return_kv_resident",
    "return_kv_restored",
    "return_kv_recomputed",
    "pauses_preserve",
    "pauses_offload",
    "pauses_release",
    "mean_slot_wait_s",
    "mean_decision_us",
    "output_mismatches",
]


def load(root):
    trials, episodes = [], []

    for run_dir in sorted(glob.glob(os.path.join(root, "*"))):
        summary_path = os.path.join(run_dir, "summary_trials.csv")
        if not os.path.exists(summary_path):
            continue

        summary = pd.read_csv(summary_path)
        summary["run"] = os.path.basename(run_dir)
        trials.append(summary)

        for path in glob.glob(os.path.join(run_dir, "trial_*_episodes.csv")):
            frame = pd.read_csv(path)
            frame["policy"] = summary["policy"].iloc[0]
            frame["arrival_rate"] = summary["arrival_rate"].iloc[0]
            frame["run"] = os.path.basename(run_dir)
            episodes.append(frame)

    if not trials:
        raise SystemExit(f"No runs found under {root}")

    return pd.concat(trials, ignore_index=True), pd.concat(episodes, ignore_index=True)


def summarize(trials):
    grouped = trials.groupby(["policy", "arrival_rate"])
    mean = grouped[METRICS].mean()
    std = grouped[["makespan_s", "p95_completion_s"]].std().add_suffix("_std")
    count = grouped.size().rename("trials")
    return pd.concat([count, mean, std], axis=1).reset_index()


def fairness(episodes):
    # Long episodes = the third with the most steps and largest contexts.
    size = episodes["steps"] * episodes["final_context_tokens"]
    cutoff = size.groupby(episodes["run"]).transform(lambda s: s.quantile(2 / 3))
    long_eps = episodes[size >= cutoff]

    return (
        long_eps.groupby(["policy", "arrival_rate"])
        .agg(
            long_episodes=("non_tool_s", "size"),
            long_mean_non_tool_s=("non_tool_s", "mean"),
            long_p95_non_tool_s=("non_tool_s", lambda s: s.quantile(0.95)),
            long_mean_slot_wait_s=("slot_wait_s", "mean"),
        )
        .reset_index()
    )


def plot(summary, column, ylabel, title, path):
    plt.figure()
    for policy, group in summary.groupby("policy"):
        group = group.sort_values("arrival_rate")
        plt.plot(group["arrival_rate"], group[column], marker="o", label=policy)
    plt.xlabel("Arrival rate (episodes/second)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("results_root")
    args = parser.parse_args()

    trials, episodes = load(args.results_root)
    summary = summarize(trials)
    fair = fairness(episodes)

    summary.to_csv(os.path.join(args.results_root, "policy_summary.csv"), index=False)
    fair.to_csv(os.path.join(args.results_root, "fairness.csv"), index=False)

    plot(
        summary,
        "p95_completion_s",
        "p95 episode completion time (s)",
        "Tail Completion Time vs Arrival Rate",
        os.path.join(args.results_root, "p95_completion_by_rate.png"),
    )
    plot(
        summary,
        "computed_tokens",
        "Prompt tokens computed per trial",
        "Prefill Work vs Arrival Rate",
        os.path.join(args.results_root, "computed_tokens_by_rate.png"),
    )

    columns = [
        "policy", "arrival_rate", "trials", "makespan_s", "p50_completion_s",
        "p95_completion_s", "p95_completion_s_std", "mean_non_tool_s",
        "computed_tokens", "return_kv_resident", "return_kv_restored",
        "return_kv_recomputed", "mean_slot_wait_s", "output_mismatches",
    ]
    with pd.option_context("display.width", 250):
        print(summary[columns].round(2).to_string(index=False))
        print()
        print(fair.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
