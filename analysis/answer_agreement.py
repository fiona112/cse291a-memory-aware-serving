"""
Check that KV policies do not change what the ReAct agent answers.

Every policy sees the same questions, prompts, and tool results at
temperature 0, so in principle each question gets the same answer. For
each policy, this reports accuracy and how often its answer to a question
matches the baseline policy's answer to the same question in the same
trial.

Usage (from the repo root, so src is importable):
  python -m analysis.answer_agreement results/react/test --baseline release
"""

import argparse
import glob
import os

import pandas as pd

from src.react_agent import normalize_answer


def load(root):
    frames = []
    for run_dir in sorted(glob.glob(os.path.join(root, "*"))):
        summary_path = os.path.join(run_dir, "summary_trials.csv")
        if not os.path.exists(summary_path):
            continue
        summary = pd.read_csv(summary_path)
        for path in glob.glob(os.path.join(run_dir, "trial_*_answers.csv")):
            answers = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
            answers["policy"] = summary["policy"].iloc[0]
            answers["arrival_rate"] = summary["arrival_rate"].iloc[0]
            frames.append(answers)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("results_root")
    parser.add_argument("--baseline", default="release")
    args = parser.parse_args()

    answers = load(args.results_root)
    answers["normalized"] = answers["answer"].astype(str).map(normalize_answer)

    baseline = answers[answers["policy"] == args.baseline][
        ["arrival_rate", "trial", "task_id", "normalized"]
    ].rename(columns={"normalized": "baseline_answer"})

    merged = answers.merge(baseline, on=["arrival_rate", "trial", "task_id"])
    merged["agrees"] = merged["normalized"] == merged["baseline_answer"]

    table = (
        merged.groupby(["arrival_rate", "policy"])
        .agg(
            answers=("agrees", "size"),
            agreement_with_baseline=("agrees", "mean"),
            exact_match=("exact_match", "mean"),
            f1=("f1", "mean"),
            finished=("finished", lambda s: s.astype(str).eq("True").mean()),
        )
        .round(3)
        .reset_index()
    )

    output = os.path.join(args.results_root, "answer_agreement.csv")
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
