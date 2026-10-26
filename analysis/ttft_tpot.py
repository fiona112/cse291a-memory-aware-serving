"""
Derive TTFT and TPOT from the per-request logs of a finished run.

llama-server reports prefill time (prompt_ms) and decode time
(predicted_ms) for each request. From those:
  TTFT (server)  = prefill + one decode step
  TPOT           = decode time per output token
  server wait    = client latency - prefill - decode
                   (time queued inside llama-server, plus HTTP overhead)

Usage:
  python analysis/ttft_tpot.py --results-dir results/runs/full_cold_oct20_01
"""

import argparse
import glob
import os

import pandas as pd


def load_requests(results_dir):
    frames = []

    for path in glob.glob(
        os.path.join(results_dir, "ctx_*_concurrency_*_trial_*_requests.csv")
    ):
        parts = os.path.basename(path).split("_")
        frame = pd.read_csv(path)
        frame["context_repeats"] = int(parts[1])
        frame["concurrency"] = int(parts[3])
        frames.append(frame)

    if not frames:
        raise SystemExit(f"No request CSVs found in {results_dir}")

    return pd.concat(frames, ignore_index=True)


def add_latency_columns(requests):
    ok = requests[
        requests["success"] & (requests["completion_tokens"] > 0)
    ].copy()

    ok["tpot_ms"] = ok["decode_ms"] / ok["completion_tokens"]
    ok["ttft_ms"] = ok["prefill_ms"] + ok["tpot_ms"]
    ok["server_wait_ms"] = (
        ok["latency"] * 1000 - ok["prefill_ms"] - ok["decode_ms"]
    ).clip(lower=0)

    return ok


def summarize(requests):
    return (
        requests.groupby("concurrency")
        .agg(
            requests=("ttft_ms", "count"),
            ttft_p50_ms=("ttft_ms", "median"),
            ttft_p95_ms=("ttft_ms", lambda s: s.quantile(0.95)),
            tpot_p50_ms=("tpot_ms", "median"),
            tpot_p95_ms=("tpot_ms", lambda s: s.quantile(0.95)),
            server_wait_p50_ms=("server_wait_ms", "median"),
        )
        .round(1)
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    requests = add_latency_columns(load_requests(args.results_dir))
    summary = summarize(requests)

    output = os.path.join(args.results_dir, "ttft_tpot_summary.csv")
    summary.to_csv(output, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
