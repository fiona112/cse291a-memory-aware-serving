"""
Why does tail-aware reclaim help? Compares min-waste runs with and without
--tail-aware on the same workload.

For every tool call (pause) it links the predicted tool time, the actual
tool time, and whether the agent's KV was reclaimed while it was away
(its next step found the KV "restored" from disk instead of "resident").
It reports, by actual tool duration:
  - how often the predictor underestimated the call (actual > 2x predicted)
  - how often the slot was reclaimed (freed for others) during the call
plus the slot wait other agents experienced.

Usage:
  python analysis/explain_tail.py \
      results/replay/test/min_waste_rate0.6 \
      results/replay/tail_test/min_waste_tailaware_rate0.6
"""

import argparse
import glob
import os

import pandas as pd

BINS = [0, 1, 5, 30.001]
LABELS = ["< 1 s", "1-5 s", "5-30 s"]


def pauses(run_dir):
    frames = []
    for path in glob.glob(os.path.join(run_dir, "trial_*_calls.csv")):
        calls = pd.read_csv(path)
        nxt = calls[["trial", "episode_id", "step", "kv_source"]].copy()
        nxt["step"] -= 1
        merged = calls.merge(
            nxt,
            on=["trial", "episode_id", "step"],
            suffixes=("", "_on_return"),
        )
        frames.append(merged[merged["pause_action"] != "finish"])

    data = pd.concat(frames, ignore_index=True)
    data["reclaimed"] = data["kv_source_on_return"] == "restored"
    data["underestimated"] = data["tool_s"] > 2 * data["predicted_tool_s"]
    data["tool_bin"] = pd.cut(data["tool_s"], BINS, labels=LABELS, right=False)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out", default="results/replay/tail_explanation.csv")
    args = parser.parse_args()

    rows = []
    for run_dir in args.run_dirs:
        summary = pd.read_csv(os.path.join(run_dir, "summary_trials.csv"))
        label = summary["policy"].iloc[0]
        rate = summary["arrival_rate"].iloc[0]
        data = pauses(run_dir)

        for tool_bin, group in data.groupby("tool_bin", observed=True):
            rows.append(
                {
                    "arrival_rate": rate,
                    "policy": label,
                    "actual_tool_time": tool_bin,
                    "pauses": len(group),
                    "share_of_tool_time": group["tool_s"].sum() / data["tool_s"].sum(),
                    "underestimated_2x": group["underestimated"].mean(),
                    "median_predicted_s": group["predicted_tool_s"].median(),
                    "median_actual_s": group["tool_s"].median(),
                    "reclaimed": group["reclaimed"].mean(),
                }
            )
        rows.append(
            {
                "arrival_rate": rate,
                "policy": label,
                "actual_tool_time": "all",
                "pauses": len(data),
                "share_of_tool_time": 1.0,
                "underestimated_2x": data["underestimated"].mean(),
                "median_predicted_s": data["predicted_tool_s"].median(),
                "median_actual_s": data["tool_s"].median(),
                "reclaimed": data["reclaimed"].mean(),
                "mean_slot_wait_s": summary["mean_slot_wait_s"].mean(),
                "mean_non_tool_s": summary["mean_non_tool_s"].mean(),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200):
        print(table.round(3).to_string(index=False))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
