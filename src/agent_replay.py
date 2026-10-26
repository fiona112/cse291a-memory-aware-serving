"""
Replay scaled TraceLab agent episodes against llama-server under a KV
policy (see src/kv_manager.py).

Each episode is an agent: at every step it sends its whole history (as
token IDs, so the server can reuse cached KV exactly), generates exactly
the traced number of output tokens (ignore_eos), then pauses for the
traced tool time while the policy decides what happens to its KV.
Episodes arrive as a Poisson process.

Start llama-server with --slot-save-path DIR (and --cache-ram 0), then:
  python -m src.agent_replay --policy offload --server-url http://127.0.0.1:8090 \
      --save-dir DIR --arrival-rate 0.3 --trials 3
"""

import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path

import aiohttp

from src.kv_manager import (
    AgentKV,
    CostModel,
    SlotPool,
    ToolTimePredictor,
    make_policy,
)
from src.metrics import percentile

FILLER_WORDS = (
    "the city has parks and schools with many stores public transport "
    "restaurants libraries homes streets people work play read write code "
    "file test build run check fix update data model value result error "
    "output input system user agent tool call return list item page line "
    "word number time place group team plan step task goal note report"
).split()


class LlamaServer:
    def __init__(self, base_url, session):
        self.base = base_url.rstrip("/")
        self.session = session

    async def post(self, path, body=None):
        async with self.session.post(
            f"{self.base}{path}", json=body or {}
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def single_token_ids(self):
        ids = []
        for word in FILLER_WORDS:
            tokens = (await self.post("/tokenize", {"content": " " + word}))["tokens"]
            if len(tokens) == 1:
                ids.append(tokens[0])
        return sorted(set(ids))

    async def slot_count(self):
        async with self.session.get(f"{self.base}/slots") as response:
            response.raise_for_status()
            return len(await response.json())

    async def complete(self, prompt_ids, slot_id, n_predict):
        return await self.post(
            "/completion",
            {
                "prompt": prompt_ids,
                "n_predict": n_predict,
                "id_slot": slot_id,
                "cache_prompt": True,
                "ignore_eos": True,
                "return_tokens": True,
                "temperature": 0,
            },
        )

    async def slot_action(self, slot_id, action, filename=None):
        body = {"filename": filename} if filename else None
        return await self.post(f"/slots/{slot_id}?action={action}", body)


class Replay:
    def __init__(self, server, pool, policy, predictor, filler_ids, args, trial):
        self.server = server
        self.pool = pool
        self.policy = policy
        self.predictor = predictor
        self.filler_ids = filler_ids
        self.args = args
        self.trial = trial
        self.nonce = uuid.uuid4().hex[:8]
        self.calls = []
        self.episodes = []
        self.saved_files = []
        self.admission = (
            asyncio.Semaphore(args.max_active) if args.max_active else None
        )
        self.start = None

    def filler(self, rng, count):
        return [rng.choice(self.filler_ids) for _ in range(max(0, count))]

    def first_prompt(self, rng, target):
        # All agents share the same system prefix (like a coding agent's
        # system prompt); the rest is unique to this episode and trial.
        shared_rng = random.Random("shared-system-prefix")
        shared = self.filler(
            shared_rng,
            min(self.args.shared_prefix, max(0, target - 16)),
        )
        return shared + self.filler(rng, target - len(shared))

    async def prepare_slot(self, agent, slot, row):
        if slot.holder == agent.agent_id:
            row["kv_source"] = "resident"
        elif agent.disk_file:
            begin = time.perf_counter()
            await self.server.slot_action(slot.slot_id, "restore", agent.disk_file)
            row["kv_source"] = "restored"
            row["restore_s"] = time.perf_counter() - begin
        else:
            row["kv_source"] = "recompute"

        slot.holder = agent.agent_id

    async def pause(self, agent, slot, context_tokens, tool_s, row):
        predicted = self.predictor.predict(agent)
        begin = time.perf_counter()
        decision = self.policy.decide(agent, context_tokens, self.pool, predicted)
        row["decision_us"] = (time.perf_counter() - begin) * 1e6
        row["pause_action"] = decision.action
        row["predicted_tool_s"] = predicted
        row["decision_reason"] = decision.reason

        if decision.action == "preserve":
            self.pool.reserve(agent, slot, ttl=decision.ttl)
            row["ttl_s"] = decision.ttl
        elif decision.action == "offload":
            filename = f"{self.nonce}_{agent.agent_id}_{row['step']}.bin"
            begin = time.perf_counter()
            await self.server.slot_action(slot.slot_id, "save", filename)
            row["save_s"] = time.perf_counter() - begin
            agent.disk_file = filename
            self.saved_files.append(filename)
            self.pool.release(slot)
        else:
            self.pool.release(slot)

        await asyncio.sleep(tool_s)
        self.predictor.record(agent, tool_s)

    async def run_episode(self, episode, arrival_s):
        await asyncio.sleep(arrival_s)
        arrived = time.perf_counter()

        if self.admission:
            await self.admission.acquire()
        admitted = time.perf_counter()

        agent = AgentKV(f"{self.nonce}-{episode['episode_id']}")
        rng = random.Random(f"{self.nonce}-{episode['episode_id']}")
        history = []
        recomputed = 0
        slot_wait = 0.0

        try:
            for index, step in enumerate(episode["steps"]):
                if index == 0:
                    history = self.first_prompt(rng, step["prompt_tokens"])
                else:
                    history += self.filler(rng, step["prompt_tokens"] - len(history))

                row = {
                    "trial": self.trial,
                    "episode_id": episode["episode_id"],
                    "step": index,
                    "prompt_tokens": len(history),
                    "target_output": step["output_tokens"],
                }

                begin = time.perf_counter()
                slot = await self.pool.acquire(agent, returning=index > 0)
                row["slot_wait_s"] = time.perf_counter() - begin
                row["slot"] = slot.slot_id
                slot_wait += row["slot_wait_s"]

                await self.prepare_slot(agent, slot, row)

                begin = time.perf_counter()
                result = await self.server.complete(
                    history,
                    slot.slot_id,
                    step["output_tokens"],
                )
                row["request_s"] = time.perf_counter() - begin
                timings = result["timings"]
                row["cached_tokens"] = timings["cache_n"]
                row["computed_tokens"] = timings["prompt_n"]
                row["prefill_ms"] = timings["prompt_ms"]
                row["decode_ms"] = timings["predicted_ms"]
                row["output_tokens"] = timings["predicted_n"]
                recomputed += timings["prompt_n"]

                history += result["tokens"]
                agent.disk_file = None  # any saved copy is now stale

                is_last = index == len(episode["steps"]) - 1
                if is_last:
                    row["pause_action"] = "finish"
                    self.pool.release(slot)
                    slot.holder = None
                else:
                    row["tool_s"] = step["tool_seconds"]
                    await self.pause(
                        agent,
                        slot,
                        len(history),
                        step["tool_seconds"],
                        row,
                    )

                self.calls.append(row)

        finally:
            if self.admission:
                self.admission.release()

        finished = time.perf_counter()
        tool_total = sum(s["tool_seconds"] for s in episode["steps"])
        self.episodes.append(
            {
                "trial": self.trial,
                "episode_id": episode["episode_id"],
                "arrival_s": arrived - self.start,
                "admission_wait_s": admitted - arrived,
                "completion_s": finished - arrived,
                "tool_s": tool_total,
                "non_tool_s": finished - arrived - tool_total,
                "slot_wait_s": slot_wait,
                "steps": len(episode["steps"]),
                "computed_tokens": recomputed,
                "final_context_tokens": len(history),
            }
        )

    async def run(self, episodes, arrival_rate, seed):
        rng = random.Random(seed)
        arrivals = []
        clock = 0.0
        for _ in episodes:
            arrivals.append(clock)
            clock += rng.expovariate(arrival_rate) if arrival_rate else 0.0

        self.start = time.perf_counter()
        await asyncio.gather(
            *(
                self.run_episode(episode, arrival)
                for episode, arrival in zip(episodes, arrivals)
            )
        )
        return time.perf_counter() - self.start


def summarize(policy_name, args, trial, makespan, episodes, calls):
    completion = [e["completion_s"] for e in episodes]
    non_tool = [e["non_tool_s"] for e in episodes]
    pauses = [c for c in calls if c["pause_action"] != "finish"]
    returns = [c for c in calls if c["step"] > 0]
    actions = [c["pause_action"] for c in pauses]

    return {
        "policy": policy_name,
        "arrival_rate": args.arrival_rate,
        "max_active": args.max_active or 0,
        "trial": trial,
        "episodes": len(episodes),
        "llm_calls": len(calls),
        "makespan_s": makespan,
        "throughput_eps": len(episodes) / makespan,
        "mean_completion_s": statistics.mean(completion),
        "p50_completion_s": percentile(completion, 50),
        "p95_completion_s": percentile(completion, 95),
        "p99_completion_s": percentile(completion, 99),
        "mean_non_tool_s": statistics.mean(non_tool),
        "p95_non_tool_s": percentile(non_tool, 95),
        "computed_tokens": sum(c["computed_tokens"] for c in calls),
        "return_kv_resident": sum(c["kv_source"] == "resident" for c in returns),
        "return_kv_restored": sum(c["kv_source"] == "restored" for c in returns),
        "return_kv_recomputed": sum(c["kv_source"] == "recompute" for c in returns),
        "pauses_preserve": actions.count("preserve"),
        "pauses_offload": actions.count("offload"),
        "pauses_release": actions.count("release"),
        "mean_slot_wait_s": statistics.mean(c["slot_wait_s"] for c in calls),
        "mean_decision_us": statistics.mean(
            c["decision_us"] for c in pauses
        ) if pauses else 0.0,
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "output_mismatches": sum(
            c["output_tokens"] != c["target_output"] for c in calls
        ),
    }


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--policy", required=True,
                        choices=["preserve", "release", "offload", "ttl", "min_waste"])
    parser.add_argument("--workload", default="workloads/tracelab_episodes.json")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--server-url", default="http://127.0.0.1:8090")
    parser.add_argument("--save-dir", required=True,
                        help="The server's --slot-save-path, to clean up saved KV files.")
    parser.add_argument("--arrival-rate", type=float, default=0.3,
                        help="Episodes per second (Poisson); 0 = all at once.")
    parser.add_argument("--max-active", type=int, default=0,
                        help="Admit at most this many episodes at once (0 = no limit).")
    parser.add_argument("--ready-priority", action="store_true")
    parser.add_argument("--ttl", type=float, default=2.0, help="TTL policy: seconds.")
    parser.add_argument("--shared-prefix", type=int, default=512)
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

    with open(args.workload, encoding="utf-8") as f:
        workload = json.load(f)
    episodes = [
        e for e in workload["episodes"]
        if args.split == "all" or e["split"] == args.split
    ]

    label = f"{args.policy}" + (f"_ttl{args.ttl:g}" if args.policy == "ttl" else "")
    if args.ready_priority:
        label += "_ready"
    if args.max_active:
        label += f"_k{args.max_active}"
    output_dir = args.output_dir or Path("results") / "replay" / (
        f"{label}_rate{args.arrival_rate:g}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    cost = CostModel(args.recompute_ms, args.save_ms, args.restore_ms)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    summaries = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        server = LlamaServer(args.server_url, session)
        n_slots = await server.slot_count()
        filler_ids = await server.single_token_ids()

        (output_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "started_at": datetime.now().astimezone().isoformat(),
                    "label": label,
                    "server_slots": n_slots,
                    "episodes": len(episodes),
                    "workload_source": workload["source"],
                    "args": {k: str(v) for k, v in vars(args).items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        for trial in range(1, args.trials + 1):
            for slot_id in range(n_slots):
                await server.slot_action(slot_id, "erase")

            replay = Replay(
                server,
                SlotPool(n_slots, ready_priority=args.ready_priority),
                make_policy(args.policy, cost, ttl_s=args.ttl),
                ToolTimePredictor(),
                filler_ids,
                args,
                trial,
            )
            makespan = await replay.run(episodes, args.arrival_rate, args.seed + trial)

            for filename in replay.saved_files:
                try:
                    os.remove(os.path.join(args.save_dir, filename))
                except OSError:
                    pass

            summary = summarize(label, args, trial, makespan, replay.episodes, replay.calls)
            summaries.append(summary)
            write_csv(output_dir / f"trial_{trial}_calls.csv", replay.calls)
            write_csv(output_dir / f"trial_{trial}_episodes.csv", replay.episodes)
            write_csv(output_dir / "summary_trials.csv", summaries)

            print(
                f"{label} trial {trial}: makespan {makespan:.1f}s, "
                f"p50 {summary['p50_completion_s']:.1f}s, "
                f"p95 {summary['p95_completion_s']:.1f}s, "
                f"computed tokens {summary['computed_tokens']}, "
                f"returns resident/restored/recomputed "
                f"{summary['return_kv_resident']}/{summary['return_kv_restored']}/"
                f"{summary['return_kv_recomputed']}, "
                f"output mismatches {summary['output_mismatches']}"
            )

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
