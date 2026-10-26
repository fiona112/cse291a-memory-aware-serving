"""
Measure what each KV-cache option costs on a running llama-server,
as a function of context length.

For every prompt length it measures, on one pinned slot:
  recompute  - prefill the whole prompt with caching disabled
  preserve   - resend prompt + a short suffix while the KV is still
               in the slot (only the suffix is prefilled)
  save       - write the slot's KV to disk (--slot-save-path)
  restore    - read it back into an erased slot, then resend
               prompt + suffix and check the cache was reused
  host_cache - (only if the server has --cache-ram > 0) push the
               prompt out of the slot with other prompts, resend it,
               and see how much came back from the CPU-RAM cache

Usage (server must be started with --slot-save-path, and each slot must
hold the longest length tested):
  python experiments/kv_cost_probe.py --label f16 --out results/kv_cost
"""

import argparse
import csv
import json
import os
import statistics
import time
import urllib.request
import uuid
from pathlib import Path

FILLER = (
    "The fictional city has parks, schools, stores, public "
    "transportation, restaurants, libraries, and neighborhoods. "
)

SUFFIX = " Summarize the text above in one sentence."


def post(url, body=None):
    data = json.dumps(body or {}).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


class Server:
    def __init__(self, base_url):
        self.base = base_url.rstrip("/")

    def count_tokens(self, text):
        return len(post(f"{self.base}/tokenize", {"content": text})["tokens"])

    def complete(self, prompt, slot, cache_prompt=True, n_predict=1):
        start = time.perf_counter()
        result = post(
            f"{self.base}/completion",
            {
                "prompt": prompt,
                "n_predict": n_predict,
                "id_slot": slot,
                "cache_prompt": cache_prompt,
                "temperature": 0,
            },
        )
        result["wall_ms"] = (time.perf_counter() - start) * 1000
        return result

    def slot_action(self, slot, action, filename=None):
        body = {"filename": filename} if filename else None
        start = time.perf_counter()
        result = post(f"{self.base}/slots/{slot}?action={action}", body)
        result["wall_ms"] = (time.perf_counter() - start) * 1000
        return result

    def slot_count(self):
        return len(get(f"{self.base}/slots"))


def make_prompt(server, target_tokens):
    # A unique prefix makes every prompt distinct, so nothing can be
    # reused from an earlier measurement.
    nonce = f"[{uuid.uuid4().hex}] "
    per_repeat = server.count_tokens(FILLER)
    repeats = max(1, target_tokens // per_repeat)
    prompt = nonce + FILLER * repeats

    while server.count_tokens(prompt) < target_tokens:
        prompt += FILLER

    return prompt


def measure_once(server, length, slot, save_dir, host_cache):
    prompt = make_prompt(server, length)
    row = {"target_tokens": length}

    # Recompute: whole prompt, no cache, on a freshly erased slot.
    server.slot_action(slot, "erase")
    recompute = server.complete(prompt, slot, cache_prompt=False)
    row["prompt_tokens"] = recompute["timings"]["prompt_n"]
    row["recompute_ms"] = recompute["timings"]["prompt_ms"]

    # Fill the slot with the prompt so its KV is resident.
    server.slot_action(slot, "erase")
    server.complete(prompt, slot)

    # Preserve: the KV is still in the slot, only the suffix is new.
    preserve = server.complete(prompt + SUFFIX, slot)
    row["preserve_ms"] = preserve["timings"]["prompt_ms"]
    row["preserve_cached"] = preserve["timings"]["cache_n"]

    # Save to disk, erase, restore, then reuse.
    filename = f"probe_{uuid.uuid4().hex}.bin"
    saved = server.slot_action(slot, "save", filename)
    row["save_ms"] = saved["timings"]["save_ms"]
    row["save_wall_ms"] = saved["wall_ms"]
    row["saved_tokens"] = saved["n_saved"]
    row["file_bytes"] = saved["n_written"]

    server.slot_action(slot, "erase")
    restored = server.slot_action(slot, "restore", filename)
    row["restore_ms"] = restored["timings"]["restore_ms"]
    row["restore_wall_ms"] = restored["wall_ms"]

    after_restore = server.complete(prompt + SUFFIX, slot)
    row["after_restore_prefill_ms"] = after_restore["timings"]["prompt_ms"]
    row["after_restore_cached"] = after_restore["timings"]["cache_n"]

    if save_dir:
        try:
            os.remove(Path(save_dir) / filename)
        except OSError:
            pass

    if host_cache:
        # Push the prompt out of the slot with a different prompt;
        # with --cache-ram the server keeps the old state in RAM.
        server.slot_action(slot, "erase")
        server.complete(prompt, slot)
        server.complete(make_prompt(server, length), slot)
        returned = server.complete(prompt + SUFFIX, slot)
        row["host_cache_prefill_ms"] = returned["timings"]["prompt_ms"]
        row["host_cache_cached"] = returned["timings"]["cache_n"]
        row["host_cache_wall_ms"] = returned["wall_ms"]

    server.slot_action(slot, "erase")
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--server", default="http://127.0.0.1:8090")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096, 6144],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/kv_cost"))
    parser.add_argument(
        "--save-dir",
        help="The server's --slot-save-path, so probe files can be deleted.",
    )
    parser.add_argument(
        "--host-cache",
        action="store_true",
        help="Also measure the CPU-RAM prompt cache (--cache-ram > 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    server = Server(args.server)

    if server.slot_count() < 1:
        raise SystemExit("llama-server reports no slots")

    # Warm up kernels so the first measurement is not a cold start.
    for _ in range(2):
        server.complete(make_prompt(server, 512), 0, cache_prompt=False)

    rows = []
    for length in args.lengths:
        for repeat in range(1, args.repeats + 1):
            row = measure_once(
                server,
                length,
                slot=0,
                save_dir=args.save_dir,
                host_cache=args.host_cache,
            )
            row["label"] = args.label
            row["repeat"] = repeat
            rows.append(row)
            print(
                f"{args.label} {length:>5} tok #{repeat}: "
                f"recompute={row['recompute_ms']:.0f}ms "
                f"preserve={row['preserve_ms']:.0f}ms "
                f"save={row['save_ms']:.0f}ms "
                f"restore={row['restore_ms']:.0f}ms "
                f"(reused {row['after_restore_cached']}/{row['prompt_tokens']})"
                + (
                    f" host_cache reused {row['host_cache_cached']}"
                    if args.host_cache else ""
                )
            )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"kv_cost_{args.label}.csv"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")

    print("\nMedian by length (ms):")
    for length in args.lengths:
        subset = [r for r in rows if r["target_tokens"] == length]
        med = lambda key: statistics.median(r[key] for r in subset)
        print(
            f"{length:>5}: recompute {med('recompute_ms'):7.1f}  "
            f"preserve {med('preserve_ms'):6.1f}  "
            f"save {med('save_ms'):6.1f}  restore {med('restore_ms'):6.1f}  "
            f"file {med('file_bytes') / 2**20:6.1f} MiB"
        )


if __name__ == "__main__":
    main()
