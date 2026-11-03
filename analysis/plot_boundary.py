"""
Summarize where our policy helps: % change in mean non-tool time vs. the
default (release) policy, with 95% bootstrap CIs, for each condition under
results/replay/boundary/<condition>/ (written by analysis/significance.py
as significance_vs_release.csv in each condition folder).

Usage:
  python analysis/plot_boundary.py results/replay/boundary
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["slots2", "slots4", "slots8", "tools0.3x", "slow_half_cpu"]
NAMES = {
    "slots2": "2 slots",
    "slots4": "4 slots",
    "slots8": "8 slots",
    "tools0.3x": "4 slots,\ntools x0.3",
    "slow_half_cpu": "4 slots,\nhalf model on CPU",
}


def load(root):
    frames = []
    for path in glob.glob(os.path.join(root, "*", "significance_vs_release.csv")):
        frame = pd.read_csv(path)
        frame["condition"] = os.path.basename(os.path.dirname(path))
        frames.append(frame.dropna(subset=["diff_pct"]))
    table = pd.concat(frames, ignore_index=True)
    table["policy_kind"] = table["policy"].map(
        lambda p: "ours (min-waste + tail + ready)" if p.startswith("min_waste") else "offload"
    )
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("root")
    args = parser.parse_args()

    table = load(args.root)
    conditions = [c for c in ORDER if c in set(table["condition"])]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.38
    for offset, (kind, color) in zip(
        (-width / 2, width / 2),
        [("offload", "#b0b0b0"), ("ours (min-waste + tail + ready)", "#1f5faf")],
    ):
        rows = (
            table[table["policy_kind"] == kind]
            .set_index("condition")
            .reindex(conditions)
        )
        x = [i + offset for i in range(len(conditions))]
        err = [
            rows["diff_pct"] - rows["diff_pct_ci95_low"],
            rows["diff_pct_ci95_high"] - rows["diff_pct"],
        ]
        ax.bar(x, rows["diff_pct"], width, yerr=err, capsize=3, color=color, label=kind)
        for xi, value, low in zip(x, rows["diff_pct"], rows["diff_pct_ci95_low"]):
            ax.text(xi, low - 1.2, f"{value:.0f}%", ha="center", va="top", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([NAMES.get(c, c) for c in conditions])
    ax.set_ylabel("Change in mean non-tool time\nvs. llama.cpp default (%)")
    ax.set_title("Where the KV policy helps (TraceLab replay, test split, rate 0.6)")
    ax.set_ylim(min(table["diff_pct_ci95_low"]) - 8, 5)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    out = os.path.join(args.root, "boundary_summary.png")
    fig.savefig(out, dpi=200)
    table.to_csv(os.path.join(args.root, "boundary_summary.csv"), index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
