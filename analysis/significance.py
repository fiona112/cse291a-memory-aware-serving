"""
Confidence intervals for policy comparisons from src.agent_replay /
src.react_agent runs.

Trial i of every policy replays the same episodes with the same arrival
seed, so episodes are paired across policies by (trial, episode_id). For
each policy this reports, against a baseline:
  - mean non-tool time with a 95% bootstrap CI
  - the paired mean difference vs. the baseline, as seconds and percent,
    with a 95% bootstrap CI (resampling episodes)
  - the difference in median completion time with a 95% bootstrap CI
A CI that excludes zero means the difference is unlikely to be noise.

Usage:
  python analysis/significance.py results/replay/test results/replay/tail_test \
      --baseline release
Run directories are found under the given roots. If two runs share a
policy label (e.g. offload in two sessions), the later root's copy is
labelled "<policy>@<root folder>".
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

METRIC = "non_tool_s"


def load(roots):
    frames = []
    seen = set()

    for root in roots:
        for run_dir in sorted(glob.glob(os.path.join(root, "*"))):
            summary_path = os.path.join(run_dir, "summary_trials.csv")
            if not os.path.exists(summary_path):
                continue

            summary = pd.read_csv(summary_path)
            label = summary["policy"].iloc[0]
            rate = summary["arrival_rate"].iloc[0]
            if (label, rate) in seen:
                label = f"{label}@{os.path.basename(os.path.normpath(root))}"
            seen.add((label, rate))

            for path in glob.glob(os.path.join(run_dir, "trial_*_episodes.csv")):
                frame = pd.read_csv(path)
                frame["policy"] = label
                frame["arrival_rate"] = rate
                frames.append(frame)

    if not frames:
        raise SystemExit("No runs found")

    return pd.concat(frames, ignore_index=True)


def bootstrap(values, statistic, rng, n=10000):
    index = rng.integers(0, len(values), size=(n, len(values)))
    stats = np.array([statistic(values[i]) for i in index])
    return np.percentile(stats, [2.5, 97.5])


def compare(episodes, baseline, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []

    for rate, group in episodes.groupby("arrival_rate"):
        base = group[group["policy"] == baseline].set_index(["trial", "episode_id"])
        if base.empty:
            continue

        for policy, runs in group.groupby("policy"):
            runs = runs.set_index(["trial", "episode_id"])
            paired = runs[[METRIC, "completion_s"]].join(
                base[[METRIC, "completion_s"]],
                rsuffix="_base",
                how="inner",
            )
            if paired.empty:
                continue

            values = runs[METRIC].to_numpy()
            ci_low, ci_high = bootstrap(values, np.mean, rng, n_boot)

            row = {
                "arrival_rate": rate,
                "policy": policy,
                "episodes": len(values),
                "mean_non_tool_s": values.mean(),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }

            if policy != baseline:
                diff = (paired[METRIC] - paired[f"{METRIC}_base"]).to_numpy()
                d_low, d_high = bootstrap(diff, np.mean, rng, n_boot)
                base_mean = paired[f"{METRIC}_base"].mean()

                pair = paired[["completion_s", "completion_s_base"]].to_numpy()
                m_low, m_high = bootstrap(
                    pair,
                    lambda p: np.median(p[:, 0]) - np.median(p[:, 1]),
                    rng,
                    n_boot,
                )

                row.update(
                    {
                        "paired_episodes": len(diff),
                        "diff_vs_baseline_s": diff.mean(),
                        "diff_ci95_low": d_low,
                        "diff_ci95_high": d_high,
                        "diff_pct": 100 * diff.mean() / base_mean,
                        "diff_pct_ci95_low": 100 * d_low / base_mean,
                        "diff_pct_ci95_high": 100 * d_high / base_mean,
                        "median_completion_diff_s": (
                            np.median(pair[:, 0]) - np.median(pair[:, 1])
                        ),
                        "median_diff_ci95_low": m_low,
                        "median_diff_ci95_high": m_high,
                        "significant": d_high < 0 or d_low > 0,
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--baseline", default="release")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="CSV path (default: <first root>/significance_vs_<baseline>.csv)")
    args = parser.parse_args()

    episodes = load(args.roots)
    table = compare(episodes, args.baseline, args.bootstrap, args.seed)
    table = table.sort_values(["arrival_rate", "mean_non_tool_s"])

    out = args.out or os.path.join(
        args.roots[0],
        f"significance_vs_{args.baseline.replace('@', '_')}.csv",
    )
    table.to_csv(out, index=False)

    columns = [
        "arrival_rate", "policy", "episodes", "mean_non_tool_s", "ci95_low", "ci95_high",
        "diff_pct", "diff_pct_ci95_low", "diff_pct_ci95_high",
        "median_completion_diff_s", "median_diff_ci95_low", "median_diff_ci95_high", "significant",
    ]
    with pd.option_context("display.width", 250):
        print(table[[c for c in columns if c in table.columns]].round(2).to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
