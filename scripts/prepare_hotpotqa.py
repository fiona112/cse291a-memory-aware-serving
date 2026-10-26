"""
Sample HotpotQA questions for the real-agent workload.

HotpotQA (Yang et al., 2018; https://hotpotqa.github.io) is licensed
CC BY-SA 4.0. Each "distractor" question comes with 10 Wikipedia
paragraphs: 2 that support the answer and 8 distractors. The agent
searches these paragraphs with a real search tool.

Download the validation split from the Hugging Face mirror:
  https://huggingface.co/datasets/hotpotqa/hotpot_qa
  (distractor/validation-00000-of-00001.parquet)

Reading Parquet needs pyarrow, which is not a project dependency;
run this one-off step in any environment that has pandas + pyarrow:
  python scripts/prepare_hotpotqa.py validation.parquet \
      --out workloads/hotpotqa_tasks.json
"""

import argparse
import json

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("parquet")
    parser.add_argument("--out", default="workloads/hotpotqa_tasks.json")
    parser.add_argument("--questions", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_parquet(args.parquet)
    sample = data.sample(n=args.questions, random_state=args.seed)

    tasks = []
    for index, (_, row) in enumerate(sample.iterrows()):
        paragraphs = [
            {"title": title, "text": "".join(sentences).strip()}
            for title, sentences in zip(
                row["context"]["title"],
                row["context"]["sentences"],
            )
        ]
        tasks.append(
            {
                "task_id": f"hq{index:03d}",
                "split": "dev" if index % 2 == 0 else "test",
                "hotpot_id": row["id"],
                "type": row["type"],
                "question": row["question"],
                "answer": row["answer"],
                "supporting_titles": sorted(set(row["supporting_facts"]["title"])),
                "paragraphs": paragraphs,
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": {
                    "dataset": "HotpotQA distractor, validation split",
                    "url": "https://hotpotqa.github.io",
                    "license": "CC BY-SA 4.0",
                    "attribution": "Yang et al., HotpotQA, EMNLP 2018",
                    "changes": f"Random sample of {args.questions} questions (seed {args.seed}).",
                },
                "tasks": tasks,
            },
            f,
            indent=1,
            ensure_ascii=False,
        )

    print(f"Wrote {len(tasks)} tasks to {args.out}")
    print(pd.Series([t["type"] for t in tasks]).value_counts().to_dict())


if __name__ == "__main__":
    main()
