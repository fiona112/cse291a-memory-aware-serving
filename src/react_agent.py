"""
Run a real ReAct agent on HotpotQA under a KV policy.

Unlike src/agent_replay.py, nothing here is scripted: the model writes
"Thought: ... Action: search[query]" or "Action: finish[answer]", the
search tool really ranks the question's 10 Wikipedia paragraphs, and the
answer is scored with HotpotQA's exact match and F1. Tool calls also
take a realistic wall time: the search itself plus a delay sampled from
the real tool times in the TraceLab workload (the same delay for a given
question and step under every policy, so policies see identical waits).

Slot handling, KV policies, and summaries are shared with agent_replay.

  python -m src.react_agent --policy min_waste --ready-priority \
      --max-active 6 --save-dir DIR --arrival-rate 0.3 --trials 3
"""

import argparse
import asyncio
import csv
import json
import random
import re
import statistics
import string
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import aiohttp

from src.agent_replay import LlamaServer, Replay, summarize
from src.kv_manager import AgentKV, CostModel, SlotPool, ToolTimePredictor, make_policy

SYSTEM_PROMPT = """You are a research agent. Answer the question by searching a small set of documents.
Respond in this exact format, one step at a time:
Thought: <your reasoning>
Action: search[<query>]
After each search you will receive an Observation with the best matching document.
When you know the answer, respond with:
Thought: <your reasoning>
Action: finish[<short answer>]
Search at least once before finishing. Answers should be a few words: a name, date, number, or yes/no.

Example:
Question: Which magazine was started first, Arthur's Magazine or First for Women?
Thought: I need the start dates of both magazines. I will search for Arthur's Magazine first.
Action: search[Arthur's Magazine]
Observation: Arthur's Magazine: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
Thought: Arthur's Magazine started in 1844. Now I need First for Women.
Action: search[First for Women]
Observation: First for Women: First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989.
Thought: First for Women started in 1989, so Arthur's Magazine was started first.
Action: finish[Arthur's Magazine]"""

STOPWORDS = set(
    "a an the of in on at to for from by with and or is was were are be been "
    "which who whom what when where why how that this these those did does do "
    "it its as than then also both same".split()
)

ACTION = re.compile(r"Action:\s*(search|finish)\[(.*?)\]", re.IGNORECASE | re.DOTALL)


def words(text):
    return [
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in STOPWORDS
    ]


def search(paragraphs, query, already_shown, top_k=2, max_words=150):
    """
    Rank paragraphs by word overlap with the query (title words count
    double) and return the top_k, like a search tool returning a page of
    results. Paragraphs already shown rank lower on ties.
    """
    query_words = set(words(query))
    scored = []

    for index, paragraph in enumerate(paragraphs):
        title_words = set(words(paragraph["title"]))
        body_words = set(words(paragraph["text"]))
        score = 2 * len(query_words & title_words) + len(query_words & body_words)
        if index in already_shown:
            score -= 0.5
        scored.append((score, -index, index))

    results = []
    for _, _, index in sorted(scored, reverse=True)[:top_k]:
        already_shown.add(index)
        paragraph = paragraphs[index]
        text = " ".join(paragraph["text"].split()[:max_words])
        results.append(f"{paragraph['title']}: {text}")

    return "\n".join(results)


def write_csv(path, rows):
    # UTF-8 explicitly: answers contain non-ASCII names, and the Windows
    # default encoding cannot write them.
    if not rows:
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_answer(text):
    # HotpotQA's official normalization.
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction, truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


class ReactServer(LlamaServer):
    async def tokenize(self, text):
        result = await self.post(
            "/tokenize",
            {"content": text, "add_special": False, "parse_special": True},
        )
        return result["tokens"]

    async def generate(self, prompt_ids, slot_id, n_predict):
        return await self.post(
            "/completion",
            {
                "prompt": prompt_ids,
                "n_predict": n_predict,
                "id_slot": slot_id,
                "cache_prompt": True,
                "return_tokens": True,
                "temperature": 0,
                "stop": ["\nObservation:", "Observation:"],
            },
        )


class ReactReplay(Replay):
    def __init__(self, *args, tool_delays, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_delays = tool_delays
        self.answers = []

    async def run_episode(self, task, arrival_s):
        await asyncio.sleep(arrival_s)
        arrived = time.perf_counter()

        if self.admission:
            await self.admission.acquire()
        admitted = time.perf_counter()

        agent = AgentKV(f"{self.nonce}-{task['task_id']}")
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {task['question']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        history = await self.server.tokenize(prompt)
        shown = set()
        answer, finished = "", False
        computed, slot_wait, tool_total = 0, 0.0, 0.0

        try:
            for index in range(self.args.max_steps):
                row = {
                    "trial": self.trial,
                    "episode_id": task["task_id"],
                    "step": index,
                    "prompt_tokens": len(history),
                }

                begin = time.perf_counter()
                slot = await self.pool.acquire(agent, returning=index > 0)
                row["slot_wait_s"] = time.perf_counter() - begin
                row["slot"] = slot.slot_id
                slot_wait += row["slot_wait_s"]

                await self.prepare_slot(agent, slot, row)

                begin = time.perf_counter()
                result = await self.server.generate(history, slot.slot_id, self.args.max_output)
                row["request_s"] = time.perf_counter() - begin
                timings = result["timings"]
                row["cached_tokens"] = timings["cache_n"]
                row["computed_tokens"] = timings["prompt_n"]
                row["prefill_ms"] = timings["prompt_ms"]
                row["decode_ms"] = timings["predicted_ms"]
                row["output_tokens"] = timings["predicted_n"]
                row["target_output"] = timings["predicted_n"]
                computed += timings["prompt_n"]

                history += result["tokens"]
                agent.disk_file = None

                action = ACTION.search(result["content"])
                row["action"] = action.group(1).lower() if action else "none"

                is_last = index == self.args.max_steps - 1
                if action and action.group(1).lower() == "finish":
                    answer, finished = action.group(2).strip(), True
                if finished or is_last or not action:
                    if not finished and action is None:
                        answer = result["content"].strip().splitlines()[-1] if result["content"].strip() else ""
                    row["pause_action"] = "finish"
                    self.pool.release(slot)
                    slot.holder = None
                    self.calls.append(row)
                    break

                # Run the real tool, then wait out the rest of a realistic delay.
                begin = time.perf_counter()
                observation = search(task["paragraphs"], action.group(2), shown)
                search_s = time.perf_counter() - begin
                delay = random.Random(f"{task['task_id']}-{index}").choice(self.tool_delays)
                tool_s = (search_s + delay) * self.args.tool_scale
                row["tool_s"] = tool_s
                tool_total += tool_s

                observation_ids = await self.server.tokenize(
                    f"\nObservation: {observation}\n"
                )
                await self.pause(agent, slot, len(history), tool_s, row)
                history += observation_ids
                self.calls.append(row)

        finally:
            if self.admission:
                self.admission.release()

        finished_at = time.perf_counter()
        exact = float(normalize_answer(answer) == normalize_answer(task["answer"]))
        self.answers.append(
            {
                "trial": self.trial,
                "task_id": task["task_id"],
                "answer": answer,
                "gold": task["answer"],
                "finished": finished,
                "exact_match": exact,
                "f1": f1_score(answer, task["answer"]),
            }
        )
        self.episodes.append(
            {
                "trial": self.trial,
                "episode_id": task["task_id"],
                "arrival_s": arrived - self.start,
                "admission_wait_s": admitted - arrived,
                "completion_s": finished_at - arrived,
                "tool_s": tool_total,
                "non_tool_s": finished_at - arrived - tool_total,
                "slot_wait_s": slot_wait,
                "steps": sum(1 for c in self.calls if c["episode_id"] == task["task_id"]),
                "computed_tokens": computed,
                "final_context_tokens": len(history),
            }
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--policy", required=True,
                        choices=["preserve", "release", "offload", "ttl", "min_waste"])
    parser.add_argument("--tasks", default="workloads/hotpotqa_tasks.json")
    parser.add_argument("--tool-times", default="workloads/tracelab_episodes.json",
                        help="Source of realistic tool delays (capped TraceLab times).")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--server-url", default="http://127.0.0.1:8090")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--arrival-rate", type=float, default=0.3)
    parser.add_argument("--max-active", type=int, default=0)
    parser.add_argument("--ready-priority", action="store_true")
    parser.add_argument("--ttl", type=float, default=1.0)
    parser.add_argument("--tool-scale", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-output", type=int, default=160)
    parser.add_argument("--recompute-ms", type=float, default=0.14)
    parser.add_argument("--save-ms", type=float, default=0.018)
    parser.add_argument("--restore-ms", type=float, default=0.013)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


async def main():
    args = parse_args()
    args.shared_prefix = 0  # unused by the ReAct agent

    with open(args.tasks, encoding="utf-8") as f:
        task_file = json.load(f)
    tasks = [
        t for t in task_file["tasks"]
        if args.split == "all" or t["split"] == args.split
    ]
    with open(args.tool_times, encoding="utf-8") as f:
        tool_delays = [
            s["tool_seconds"]
            for e in json.load(f)["episodes"]
            for s in e["steps"]
            if s["tool_seconds"] > 0
        ]

    label = args.policy + (f"_ttl{args.ttl:g}" if args.policy == "ttl" else "")
    if args.ready_priority:
        label += "_ready"
    if args.max_active:
        label += f"_k{args.max_active}"
    output_dir = args.output_dir or Path("results") / "react" / (
        f"{label}_rate{args.arrival_rate:g}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    cost = CostModel(args.recompute_ms, args.save_ms, args.restore_ms)
    connector = aiohttp.TCPConnector(force_close=True)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    summaries = []

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        server = ReactServer(args.server_url, session)
        n_slots = await server.slot_count()

        (output_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "started_at": datetime.now().astimezone().isoformat(),
                    "label": label,
                    "server_slots": n_slots,
                    "tasks": len(tasks),
                    "task_source": task_file["source"],
                    "args": {k: str(v) for k, v in vars(args).items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        for trial in range(1, args.trials + 1):
            for slot_id in range(n_slots):
                await server.slot_action(slot_id, "erase")

            pool = SlotPool(n_slots, ready_priority=args.ready_priority)
            replay = ReactReplay(
                server,
                pool,
                make_policy(args.policy, cost, ttl_s=args.ttl),
                ToolTimePredictor(),
                [],
                args,
                trial,
                tool_delays=tool_delays,
            )
            if args.policy == "min_waste":
                pool.enable_reclaim(replay.reclaim, cost)

            makespan = await replay.run(tasks, args.arrival_rate, args.seed + trial)

            for filename in replay.saved_files:
                try:
                    Path(args.save_dir, filename).unlink()
                except OSError:
                    pass

            summary = summarize(
                label, args, trial, makespan,
                replay.episodes, replay.calls, replay.reclaims,
            )
            summary["exact_match"] = statistics.mean(a["exact_match"] for a in replay.answers)
            summary["f1"] = statistics.mean(a["f1"] for a in replay.answers)
            summary["finished_rate"] = statistics.mean(a["finished"] for a in replay.answers)
            summaries.append(summary)

            write_csv(output_dir / f"trial_{trial}_calls.csv", replay.calls)
            write_csv(output_dir / f"trial_{trial}_episodes.csv", replay.episodes)
            write_csv(output_dir / f"trial_{trial}_answers.csv", replay.answers)
            write_csv(output_dir / f"trial_{trial}_reclaims.csv", replay.reclaims)
            write_csv(output_dir / "summary_trials.csv", summaries)

            print(
                f"{label} trial {trial}: makespan {makespan:.1f}s, "
                f"p50 {summary['p50_completion_s']:.1f}s, p95 {summary['p95_completion_s']:.1f}s, "
                f"EM {summary['exact_match']:.2f}, F1 {summary['f1']:.2f}, "
                f"computed tokens {summary['computed_tokens']}"
            )

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
