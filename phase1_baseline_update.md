# Phase 1 Baseline Update

Oct 19, 2026

## Summary

On a clean llama-server, FIFO throughput rises from 0.7 tasks/s at concurrency 1 to 2.4 tasks/s at concurrency 8, and c=8 is the best fixed baseline so far. The reported 36 trials contain 1,728 tasks (3 context sizes × 4 concurrency levels × 3 trials × 48 tasks), all reported successful. The bottleneck is GPU decode, not memory.

- **What changed vs the original repo:** VRAM measurement, per-request instrumentation, tail latency (p50/p95/p99), request timeouts, a context-size sweep, CLI flags, a bigger workload (48 tasks per trial), and a fix for a process-detection bug. These changes are on branch `vram-measurement`.
- **Two findings that affect Phase 2:** llama.cpp preallocates the whole KV cache at startup, so VRAM does not move with concurrency. Identical prompts across trials hit the prefix cache almost 100%, so the context-size effect is currently hidden.
- **One run had to be thrown out:** a llama-server that had been up for hours ran 2–3x slower; restarting it fixed this.
- **Decisions needed from the team:** which AWS instance to use for final runs, how to handle the prefix cache, and whether to restart the server between trials.

## Setup

All results below come from one Windows laptop, so they are development numbers, not the final AWS numbers.

| Item | Value |
| --- | --- |
| Machine | Windows 11 laptop, NVIDIA RTX 5070 Laptop GPU (8 GB VRAM), driver 596.49 |
| Runtime | llama.cpp build b11748, CUDA 13.4 Windows release |
| Model | Qwen2.5-1.5B-Instruct, Q4_K_M GGUF (official Qwen repo) |
| Server command | `llama-server -m qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 -c 32768 -np 8 -ngl 99` |
| Slots | 8 slots × 4,096 tokens each; all layers on GPU |
| Python | 3.12.10 venv; `pip freeze` matches `requirements.txt` exactly (plus `nvidia-ml-py`) |

`-c 32768 -np 8` was chosen so every slot fits the ~3,900-token `long_context` prompt. The original server flags were not recorded in the repo, so this may differ from Fiona's setup. Python 3.12 is needed because the pinned `numpy==2.0.2` does not install on 3.13.

## Changes vs the original code

Every change adds to the original pipeline; existing CSV columns and plots are kept, and running with no flags reproduces the original concurrency sweep.

| Area | Original | Now | File |
| --- | --- | --- | --- |
| Workload size | 3 tasks per type (12 per trial) | 12 per type (48 per trial), `--tasks-per-type` | `src/workload.py` |
| Trials | 3, hard-coded | 3, `--trials` | `src/run_experiment.py` |
| Context size | `long_context` fixed at 120 repeats (~3,900 tokens) | `--context-repeats 30 60 120` sweeps ~1,000 / 2,000 / 3,900 tokens | `src/workload.py` |
| Tail latency | Mean and max only | p50 / p95 / p99 completion time, new tail-latency plot | `src/metrics.py`, `analysis/analyze.py` |
| Per-request data | Whole-task time only | One row per LLM call: latency, prompt/output tokens, cached tokens, prefill ms, decode ms (`*_requests.csv`, `request_summary.csv`) | `src/client.py`, `src/run_experiment.py` |
| Timeouts | None | 120 s per request (`--timeout`); `failed_tasks`, `timed_out_tasks`, `failure_rate` | `src/client.py`, `src/metrics.py` |
| VRAM | Not measured | `peak_vram_mb` + `vram_scope` via NVML; empty on machines without NVIDIA | new `src/gpu_memory.py` |
| Output tokens | Not recorded | `output_tokens` per task, to check gains are not from shorter outputs | `src/metrics.py` |
| Process detection | Matched any process whose command line contained "llama-server" | Matches the executable name only | `src/run_experiment.py` |
| Dependencies | — | `nvidia-ml-py==13.615.71` | `requirements.txt` |

**Bug fix worth knowing:** the original `find_llama_server_process()` could return the shell that launched the server instead of the server. In one test it measured a 9 MB bash process, which silently makes RSS numbers wrong.

**VRAM scope:** on Linux (AWS) NVML reports the llama-server process's own VRAM (`vram_scope = process`). On Windows it only reports whole-GPU usage, including other apps (`vram_scope = device`).

**Output file names changed** to `results/ctx_<repeats>_concurrency_<c>_trial_<t>.csv`. `analyze.py` still reads older `summary_trials.csv` files without the new columns.

Run it with:

```
python -m src.run_experiment --context-repeats 30 60 120
python analysis/analyze.py
```

## Results

Throughput grows with concurrency at every context size, with diminishing returns: about 2x from c=1 to c=2, +40% to c=4, +15% to c=8. Trial-to-trial spread is small (standard deviation ≤ 0.1 tasks/s in 10 of 12 cells).

| Context (`long_context` prompt) | c=1 | c=2 | c=4 | c=8 |
| --- | --- | --- | --- | --- |
| ~1,000 tokens | 0.68 | 1.51 | 1.79 | 2.26 |
| ~2,000 tokens | 0.69 | 1.35 | 2.28 | 2.52 |
| ~3,900 tokens | 0.71 | 1.44 | 2.01 | 2.37 |

Throughput in tasks/s, mean of 3 trials × 48 tasks, reported from a local `results/summary_aggregated.csv` (context sweep on a fresh server, 2026-10-19). Plot: `results/throughput_vs_concurrency.png`. The trial and request CSVs, and the derived aggregate CSV, are not included in this branch, so the table and plots cannot be independently regenerated from the committed files.

Latency and decode time at the ~3,900-token context (other context sizes look the same):

| Concurrency | Mean queue (s) | p50 completion (s) | p95 completion (s) | p99 completion (s) | long_output decode per request (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | 31.2 | 32.7 | 62.1 | 66.1 | 2.2 |
| 2 | 14.7 | 15.5 | 30.2 | 32.2 | 1.8 |
| 4 | 10.7 | 12.6 | 23.1 | 23.7 | 3.5 |
| 8 | 6.4 | 8.8 | 19.4 | 20.0 | 6.7 |

No task failed or timed out in any of the 36 trials. Memory: llama-server RSS rose from about 1,900 MB to 5,800 MB over the run, while whole-GPU VRAM stayed between 3,330 and 3,480 MB in every cell (see Limitations for why).

## Conclusions

1. **The dominant bottleneck is GPU decode, not memory.** As concurrency goes from 1 to 8, the 400-token `long_output` requests take 2.2 s → 6.7 s to decode, because all slots share the GPU. Prefill is cheap by comparison: about 0.12 s per 1,000 tokens actually computed (measured on uncached prompts of up to ~2,000 tokens), so a fully uncached ~3,900-token prompt would take roughly 0.5 s.
2. **c=8 is the best fixed baseline on this machine.** It has the highest throughput (2.26–2.52 tasks/s) and the lowest p50 and p95 completion times, and no failures. Throughput is still rising at c=8, so the knee has not been found yet.
3. **Memory is not under pressure in this setup.** There are no failures or timeouts. llama.cpp reserves the whole KV cache (`-c 32768`) in VRAM at startup, so VRAM does not move with concurrency or context size.
4. **Task count per trial matters.** With 12 tasks per trial, c=4 and c=8 looked the same (about 1.6 tasks/s); with 48 tasks, c=8 is clearly ahead. Small runs are dominated by the tail of the last few slow tasks.

## Problems found and limitations

Four things limit how far these numbers can be trusted; the first was severe enough to invalidate a whole run.

**1. A long-running llama-server slows down 2–3x.** A sweep run on a server that had been up for about 2.5 hours gave c=4 throughput of about 0.6 tasks/s and `long_output` decode of about 16 s. After a restart, the same test gave 1.2–2.1 tasks/s and 2.5–6 s decode. The invalid run's raw output is not included in this branch and should not be used. We ruled out power/throttling and the new VRAM polling, but the root cause is unknown. Disabling the host prompt cache (`--cache-ram 0`) made no clear difference in a short test.

**2. The prefix cache hides the effect of context size.** The same task IDs and prompts repeat in every trial, so llama-server reuses them: `long_context` requests reused 3,903 of 3,904 prompt tokens from cache after the first trial. Context size therefore looks free. This is a benchmark artifact, not realistic agent reuse.

**3. RSS tracks run time, not the experiment variables.** llama-server RSS grew steadily from about 1,900 MB to 5,800 MB in run order (context 30, then 60, then 120). This looks like the host prompt cache filling up (llama.cpp's default `--cache-ram` is 8,192 MiB), but we have not proven it. As a result, "bigger context uses more RAM" in the plots is confounded with "the server has been running longer".

**4. VRAM is preallocated, and on Windows it is whole-GPU.** The KV cache for all 32,768 tokens is reserved at startup, so VRAM is flat at 3,330–3,480 MB. On Windows, NVML cannot report per-process VRAM, so the number also includes other apps. On Linux/AWS it will be the server's own usage.

Other caveats: all numbers are from one laptop GPU, and the server flags may differ from what the original baseline used.

## Open questions and next steps

The biggest open question is how Phase 2 should define "memory pressure", because VRAM on llama.cpp does not move at runtime.

**Decisions for the team**

- [ ] Which AWS instance for final runs (NVIDIA g4dn/g5/g6, AMD g4ad, or CPU)? It decides how memory is measured, and a large GPU may never reach memory pressure with a 1.5B model.
- [ ] Standard llama-server flags (`-c`, `-np`, `-ngl`, `--cache-ram`), written into the README.
- [ ] Prefix cache: remove the artificial cross-trial hits (for example, a unique ID at the start of each trial's prompts), keep the cache on, and test cache on vs off as its own experiment?
- [ ] Make `multistep` step 2 send the full conversation history, so the cache behaves like a real agent that retains context across a tool call?
- [ ] Restart llama-server before every trial, or at least before every run?

**Possible directions for Phase 2**

- Measure KV-cache occupancy in tokens instead of VRAM bytes, using a shared pool (`--kv-unified`) with a smaller `-c`, so concurrent long contexts actually compete for memory.
- A memory-aware admission policy: estimate each request's tokens (prompt + `max_tokens`) before admitting it. This matches the spec's "estimating the memory needed by incoming requests".

**Smaller follow-ups**

- Test c=16 and above to find where throughput stops rising.
- Randomize trial order with a fixed seed, so slow drift in the server is spread across conditions.
- Record the run environment (server flags, llama.cpp build, GPU, git commit) with each results folder.
