# cse291a-memory-aware-serving

CSE 291A course project (Option 1: Memory-Aware Agent Serving). Hyeonung Cho, Fiona Feng, Yu-Hsuan Lin.

We study how a serving system should handle an agent's KV cache while the agent waits for a tool call: keep it in its GPU slot, save it to disk and restore it later, or let it be overwritten and recompute it. Phase 1 builds a FIFO baseline and characterizes the problem; Phase 2 builds and evaluates per-pause KV policies. For a full summary of methods and results, see [`PROJECT_REPORT1027.md`](PROJECT_REPORT1027.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/run_experiment.py`, `src/workload.py`, `src/scheduler.py`, `src/client.py`, `src/metrics.py` | Phase 1: FIFO baseline with fixed concurrency, microbenchmark workload, per-task and per-request metrics |
| `src/gpu_memory.py` | VRAM measurement via NVML (empty on machines without an NVIDIA GPU) |
| `src/kv_manager.py` | Phase 2: slot pool owned by the scheduler and KV policies (preserve, release, offload, TTL, min-waste) |
| `src/agent_replay.py` | Phase 2: replays scaled TraceLab agent episodes under a KV policy |
| `src/react_agent.py` | Phase 2: real ReAct agent on HotpotQA under a KV policy |
| `experiments/kv_cost_probe.py` | Measures recompute, keep, save/restore, and CPU-cache costs vs. context length |
| `scripts/prepare_tracelab.py`, `scripts/prepare_hotpotqa.py` | Build the committed workloads from the public datasets |
| `workloads/` | `tracelab_episodes.json` (40 episodes), `hotpotqa_tasks.json` (60 questions), each split into dev/test |
| `analysis/` | Analysis and plotting scripts (see below) |
| `results/` | Raw measurements, summaries, and plots for every experiment |

## Setup

**Python.** Python 3.12 (the pinned `numpy==2.0.2` does not install on 3.13).

```
python -m venv .venv
.venv/Scripts/activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**llama.cpp.** Download a prebuilt release from <https://github.com/ggml-org/llama.cpp/releases> for your platform (we used build b11748 with CUDA 13.4 on Windows, and build 11669 with Metal on macOS).

**Model.** Qwen2.5-1.5B-Instruct, Q4_K_M, used in all comparisons:

```
curl -L -o qwen2.5-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

All scripts talk to `llama-server` over HTTP on localhost, so start the server first (commands below) and keep it running. Run Python commands from the repository root.

## Phase 1: FIFO baseline

The runner checks the server's flags and refuses to start unless it uses 8 slots, has the host prompt cache disabled, and has a slot-save directory (used to clear KV state between trials):

```
llama-server -m qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 \
  -c 32768 -np 8 -ngl 99 --cache-ram 0 --slot-save-path SLOT_DIR
```

```
# Concurrency x context-size sweep (defaults: concurrency 1-32, contexts 1k/2k/4k, 3 trials, 48 tasks per trial)
python -m src.run_experiment
python analysis/analyze.py --results-dir results/runs/<run folder>     # plots, summary_aggregated.csv, report.html

# TTFT / TPOT from the per-request logs of a run
python analysis/ttft_tpot.py --results-dir results/runs/<run folder>

# Idle periods between agent steps (one run per tool delay)
python -m src.run_experiment --concurrency 1 4 8 16 --context-repeats 90 --agent-history --tool-delay 2 \
  --output-dir results/runs/idle_sweep_delay2
python analysis/idle_sweep.py results/runs/idle_sweep_delay0.5 results/runs/idle_sweep_delay2 results/runs/idle_sweep_delay8
```

Each sweep writes `run_config.json` (server command, llama.cpp version, seed, trial order), per-trial task and request CSVs, and `summary_trials.csv` into its own folder. Trial order is shuffled with a fixed seed, and every slot is erased before each trial.

**KV cost measurement** (any port; here 8090, 4 slots of 8k tokens):

```
llama-server -m qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8090 -c 32768 -np 4 -ngl 99 --cache-ram 0 --slot-save-path SLOT_DIR
python experiments/kv_cost_probe.py --label f16 --save-dir SLOT_DIR
```

## Phase 2: per-pause KV policies

Phase 2 uses a tighter KV budget so active agents outnumber slots:

```
llama-server -m qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8090 \
  -c 16384 -np 4 -ngl 99 --cache-ram 0 --slot-save-path SLOT_DIR
```

**Trace replay** (TraceLab episodes):

```
# Our policy: min-waste with tail-aware reclaim and ready-step priority
python -m src.agent_replay --policy min_waste --tail-aware --ready-priority \
  --split test --arrival-rate 0.6 --trials 5 --save-dir SLOT_DIR

# Baselines
python -m src.agent_replay --policy release ...             # llama-server default behavior
python -m src.agent_replay --policy offload ...             # save to disk at every tool call
python -m src.agent_replay --policy preserve ...            # keep the slot reserved
python -m src.agent_replay --policy ttl --ttl 1 ...         # Continuum-style time-to-live
python -m src.agent_replay --policy release --max-active 6 ...   # fixed admission limit
```

Other options: `--split dev|test|all`, `--tool-scale` (multiply tool times), `--recompute-ms/--save-ms/--restore-ms` (per-token cost model; set from `kv_cost_probe.py` on the target machine). `--save-dir` must be the server's `--slot-save-path`.

**Real agent** (ReAct on HotpotQA; for full-page search results use `--top-k 10 --max-words 400` with 16k-token slots, `-c 65536 -np 4`):

```
python -m src.react_agent --policy min_waste --tail-aware --ready-priority \
  --split test --arrival-rate 2 --trials 3 --save-dir SLOT_DIR
```

**Analysis:**

| Script | Purpose |
| --- | --- |
| `analysis/compare_policies.py results/replay/test` | Per-policy summary, long-episode fairness, plots |
| `analysis/significance.py results/replay/test --baseline release` | Paired bootstrap 95% CIs vs. a baseline |
| `analysis/explain_tail.py <run dirs>` | Predicted vs. actual tool time and reclaim behavior |
| `analysis/plot_boundary.py results/replay/boundary` | Boundary study chart |
| `python -m analysis.answer_agreement results/react/test` | Answer agreement across policies |

## Results map

| Folder | Experiment |
| --- | --- |
| `results/runs/` | Phase 1 sweeps (team runs on macOS and Windows) and the idle-period sweep |
| `results/kv_cost/` | KV cost measurements, `summary.md` |
| `results/idle_sweep/` | KV reuse across tool calls |
| `results/replay/m3`, `m4_dev`, `tail_dev` | Development-split policy runs |
| `results/replay/test`, `tail_test` | Held-out test-split comparison |
| `results/replay/boundary/` | Slot count, tool length, and slower-compute conditions |
| `results/react/` | Real ReAct agent runs |

Summaries: `results/kv_cost/summary.md`, `results/replay/summary.md`, `results/react/summary.md`.

## Data sources

- **TraceLab** coding-agent trace dataset, SyFI Lab, University of Washington, CC BY 4.0 (<https://tracelab.cs.washington.edu>). `workloads/tracelab_episodes.json` is a scaled sample; to rebuild it, download `syfi_coding_trace.jsonl.gz` (release v0.0.2) and run `python scripts/prepare_tracelab.py <path>`.
- **HotpotQA**, Yang et al., EMNLP 2018, CC BY-SA 4.0 (<https://hotpotqa.github.io>). `workloads/hotpotqa_tasks.json` is a 60-question sample of the distractor validation split; to rebuild it, run `scripts/prepare_hotpotqa.py` on the Hugging Face parquet file (needs `pyarrow`, which is not a project dependency).
