"""
Build a small, replayable agent workload from the TraceLab dataset.

TraceLab (SyFI Lab, University of Washington, https://tracelab.cs.washington.edu)
records real Claude Code / Codex sessions, one row per LLM call ("round"),
with token counts and per-tool wall latencies. Licensed CC BY 4.0. The
dataset itself is not committed; download syfi_coding_trace.jsonl.gz from
https://github.com/uw-syfi/TraceLab/releases (v0.0.2) and pass its path.

An "episode" is a run of rounds chained by tool results: it starts at a
user message and continues while each next round is fed by tool output.
For each sampled episode we keep the first --max-rounds rounds and scale
it to fit our model and KV budget:

  prompt tokens  scaled by one factor per episode, so the final prompt is
                 --final-prompt-tokens; this keeps the "large context,
                 small appends" shape of real coding agents
  output tokens  scaled by --output-scale, clamped to
                 [--min-output, --max-output]
  tool time      the wall-clock span of the round's tool calls, capped at
                 --tool-cap seconds (the dataset's p95 is about 30 s)

Usage:
  python scripts/prepare_tracelab.py path/to/syfi_coding_trace.jsonl.gz \
      --out workloads/tracelab_episodes.json
"""

import argparse
import gzip
import json
import random
from collections import defaultdict
from datetime import datetime


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def tool_span_seconds(tools):
    # Tools in one round may overlap, so use the span from the first
    # call to the last result rather than the sum of latencies.
    spans = [
        (parse_time(t["emitted_at"]), parse_time(t["result_at"]))
        for t in tools
        if t.get("emitted_at") and t.get("result_at")
    ]
    if spans:
        start = min(s for s, _ in spans)
        end = max(e for _, e in spans)
        return max(0.0, (end - start).total_seconds())

    latencies = [
        t["tool_wall_latency_ms"]
        for t in tools
        if t.get("tool_wall_latency_ms") is not None
    ]
    return max(latencies) / 1000 if latencies else 0.0


def load_episodes(path, min_rounds):
    sessions = defaultdict(list)

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            sessions[record["session_id"]].append(
                {
                    "round_index": record["round_index"],
                    "provider": record["provider"],
                    "input_tokens": record["input_tokens_total"],
                    "output_tokens": record["output_tokens"] or 0,
                    "starts_with_user": (
                        record["first_input_event_type"] == "user_message"
                    ),
                    "tool_seconds": tool_span_seconds(record["tools"]),
                    "has_tools": bool(record["tools"]),
                }
            )

    episodes = []
    for session_id, rounds in sessions.items():
        rounds.sort(key=lambda r: r["round_index"])
        current = []

        for r in rounds:
            if r["starts_with_user"] and current:
                episodes.append((session_id, current))
                current = []
            current.append(r)
        if current:
            episodes.append((session_id, current))

    # Every round but the last must call a tool, or the chain is broken.
    return [
        (session_id, rounds)
        for session_id, rounds in episodes
        if len(rounds) >= min_rounds
        and all(r["has_tools"] for r in rounds[: min_rounds - 1])
        and rounds[0]["input_tokens"] > 0
    ]


def scale_episode(rounds, args):
    rounds = rounds[: args.max_rounds]

    # Keep only the chained prefix: stop after the first round without tools.
    for i, r in enumerate(rounds):
        if not r["has_tools"]:
            rounds = rounds[: i + 1]
            break

    factor = args.final_prompt_tokens / max(r["input_tokens"] for r in rounds)

    steps = []
    for i, r in enumerate(rounds):
        is_last = i == len(rounds) - 1
        output = round(r["output_tokens"] * args.output_scale)
        steps.append(
            {
                "prompt_tokens": max(1, round(r["input_tokens"] * factor)),
                "output_tokens": min(
                    args.max_output,
                    max(args.min_output, output),
                ),
                "tool_seconds": 0.0 if is_last else round(
                    min(args.tool_cap, r["tool_seconds"]),
                    3,
                ),
                "tool_seconds_uncapped": 0.0 if is_last else round(
                    r["tool_seconds"],
                    3,
                ),
            }
        )

    # Prompts never shrink inside an episode (each call resends history).
    for previous, step in zip(steps, steps[1:]):
        step["prompt_tokens"] = max(
            step["prompt_tokens"],
            previous["prompt_tokens"] + previous["output_tokens"],
        )

    return steps, factor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("trace", help="Path to syfi_coding_trace.jsonl.gz")
    parser.add_argument("--out", default="workloads/tracelab_episodes.json")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--min-rounds", type=int, default=6)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--final-prompt-tokens", type=int, default=3000)
    parser.add_argument("--output-scale", type=float, default=0.25)
    parser.add_argument("--min-output", type=int, default=16)
    parser.add_argument("--max-output", type=int, default=128)
    parser.add_argument("--tool-cap", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    candidates = load_episodes(args.trace, args.min_rounds)
    print(f"{len(candidates)} candidate episodes with >= {args.min_rounds} chained rounds")

    # At most one episode per session, so no single session dominates.
    by_session = {}
    for session_id, rounds in candidates:
        by_session.setdefault(session_id, rounds)

    rng = random.Random(args.seed)
    chosen = rng.sample(sorted(by_session.items()), args.episodes)

    episodes = []
    for index, (session_id, rounds) in enumerate(chosen):
        steps, factor = scale_episode(rounds, args)
        episodes.append(
            {
                "episode_id": f"ep{index:03d}",
                "split": "dev" if index % 2 == 0 else "test",
                "provider": rounds[0]["provider"],
                "source_session": session_id,
                "token_scale": round(factor, 6),
                "steps": steps,
            }
        )

    workload = {
        "source": {
            "dataset": "TraceLab public coding-agent trace (v0.0.2)",
            "url": "https://tracelab.cs.washington.edu",
            "license": "CC BY 4.0",
            "attribution": "TraceLab (SyFI Lab, University of Washington)",
            "changes": (
                "Sampled episodes, truncated to the first rounds, token "
                "counts and output lengths scaled, tool time capped; see "
                "parameters."
            ),
        },
        "parameters": vars(args) | {"trace": None},
        "episodes": episodes,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(workload, f, indent=1)

    steps = [s for e in episodes for s in e["steps"]]
    tools = sorted(s["tool_seconds"] for s in steps if s["tool_seconds"] > 0)
    print(f"Wrote {len(episodes)} episodes, {len(steps)} LLM calls to {args.out}")
    print(
        f"Tool seconds: median {tools[len(tools) // 2]:.2f}, "
        f"max {tools[-1]:.1f}, total {sum(tools):.0f}"
    )
    print(
        "Output tokens total:",
        sum(s["output_tokens"] for s in steps),
    )


if __name__ == "__main__":
    main()
