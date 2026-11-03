# Memory-Aware Agent Serving

CSE 291A: Systems for LLMs and AI Agents

This project studies how to serve multi-step LLM agents efficiently under limited compute and memory.

The project began with a fixed-concurrency FIFO serving baseline. Phase 1 showed that increasing concurrency improves throughput until the server becomes compute/decode saturated, while process memory remains relatively flat because `llama.cpp` preallocates KV-cache capacity.

Based on these findings, the project shifted in Phase 2 toward **KV-cache management during agent tool pauses**. The main question is:

> When an agent pauses for a tool call, should its KV cache stay on the GPU, be moved to CPU memory or SSD, or be discarded and recomputed later?

We evaluate several KV-management policies using controlled microbenchmarks, realistic agent traces, and a real ReAct agent workload.

---

# Phase 1: FIFO Baseline and Characterization

Phase 1 implements a fixed-concurrency FIFO serving baseline using:

- **Model:** Qwen2.5-1.5B-Instruct
- **Quantization:** Q4_K_M
- **Runtime:** `llama.cpp`
- **Scheduler:** FIFO with fixed client-side concurrency
- **Server parallelism:** 8 inference slots

The workload contains:

- short-context / short-output requests,
- long-context requests,
- long-output requests,
- multi-step tasks with controlled tool delays.

We sweep approximate context sizes of:

```text
1k, 2k, 4k tokens
```

and concurrency levels:

```text
1, 2, 4, 8, 16, 32
```

Each configuration is repeated across multiple trials with fixed-seed ordering and controlled cache state.

---

## Phase 1 Metrics

We instrument both individual model requests and complete agent tasks.

Collected metrics include:

- task completion time,
- queueing time,
- execution time,
- throughput,
- p50 / p95 / p99 completion latency,
- prompt and output token counts,
- request-level timing,
- TTFT,
- TPOT,
- peak process memory,
- VRAM when available,
- failures and timeouts.

---

## Phase 1 Findings

### Throughput improves with concurrency, then saturates

Throughput increases substantially from concurrency 1 to 16, but changes very little between 16 and 32.

For the ~1k workload:

```text
Concurrency 1:  ~0.43 tasks/s
Concurrency 16: ~1.50 tasks/s
Concurrency 32: ~1.49 tasks/s
```

The same plateau appears for larger contexts.

### Execution time continues increasing after saturation

Higher concurrency reduces FIFO queueing, but active requests increasingly compete for the same inference capacity.

For the ~4k workload, mean execution time grows from roughly:

```text
Concurrency 1:   ~3 s
Concurrency 8:  ~10.5 s
Concurrency 16: ~18 s
Concurrency 32: >31 s
```

This suggests that once the server is saturated, additional concurrency mainly creates contention rather than increasing completed work.

### Longer contexts are more expensive

Longer-context workloads consistently show:

- lower throughput,
- higher execution time,
- higher completion latency,
- higher tail latency.

### Process memory remains mostly flat

Peak process memory stays around roughly 1.9–2.0 GB across most configurations.

This is largely because `llama.cpp` preallocates KV-cache capacity, so process RSS does not track request pressure very well.

### Prefix caching materially affects serving cost

Earlier experiments reused cached prompt state across trials, which made repeated long-context requests artificially cheap.

Final Phase 1 experiments therefore clear reusable state between trials.

### No failures were observed

Across the tested configurations:

```text
Failure rate: 0%
```

---

## Phase 1 Bottleneck

The dominant bottleneck in the Phase 1 setup is **decode / compute capacity rather than physical memory exhaustion**.

Throughput largely saturates around concurrency 16 while execution time continues increasing. This means simple adaptive concurrency based only on process memory has little useful signal in the current runtime.

These results motivated the Phase 2 shift toward direct KV-cache management.

---

# Phase 2: KV-Cache Management During Agent Tool Pauses

Phase 2 studies what to do with an agent's KV cache while the agent is waiting for an external tool.

The system can choose among several actions:

- **Keep:** retain KV state on the GPU
- **CPU offload:** move KV state to CPU memory (its cost was measured, but it is not used as a policy option)
- **SSD offload:** move KV state to storage
- **Recompute:** discard KV state and rebuild it when the agent resumes

The goal is to minimize wasted compute, transfer cost, and memory occupancy while maintaining low latency.

---

## KV Cost Characterization

We added direct measurements of:

- KV size per token,
- recomputation cost,
- CPU save/restore cost,
- SSD save/restore cost,
- TTFT,
- TPOT.

These measurements provide the basic cost model used by the Phase 2 policies.

---

## Idle-Period Sweep

We added an idle-period sweep to measure how tool-call duration changes the preferred KV action.

This directly studies the central Phase 2 tradeoff:

- short pauses may favor keeping KV on the GPU,
- longer pauses may justify offloading or evicting,
- recomputation may be cheaper than storing state in some cases.

---

## Trace-Driven Evaluation

We added a scaled-down TraceLab-style agent workload with realistic:

- multi-step request sequences,
- tool timings,
- context growth,
- model pauses,
- KV reuse opportunities.

The trace replay framework evaluates KV policies at each pause in the agent trajectory.

---

## KV Policies

Implemented policies include:

- always keep (`preserve`),
- release: free the slot and recompute only if it was overwritten (`release`, llama-server's default behavior),
- always offload to SSD (`offload`),
- a fixed admission limit (`--max-active`),
- on-demand min-waste,
- TTL-based policies,
- tail-aware min-waste.

The min-waste family chooses an action using estimated costs for:

- memory occupancy,
- recomputation,
- transfer,
- expected pause duration,
- future reuse.

TTL parameters are tuned on a development split and evaluated on a held-out test split.

---

## Real Agent Evaluation

We also added a real **ReAct agent on HotpotQA** and evaluate it under the KV-management policies.

This complements the controlled microbenchmarks and trace replay with an actual end-to-end tool-using agent workload.

The full evaluation therefore spans three levels:

1. controlled microbenchmarks,
2. trace-driven replay,
3. real agent execution.

---

## Tail-Aware Min-Waste

The current strongest policy is a **tail-aware min-waste reclaim policy**.

This policy was selected using development data and then evaluated on a held-out test split.

Rather than optimizing only average cost, it also accounts for tail behavior when deciding which KV state to retain or reclaim.

---

# Repository Structure

```text
.
├── analysis/          # analysis and plotting scripts (Phase 1 and 2)
├── experiments/       # kv_cost_probe.py
├── results/           # raw measurements, summaries, plots
├── scripts/           # prepare_tracelab.py, prepare_hotpotqa.py
├── src/
│   ├── client.py, gpu_memory.py, metrics.py
│   ├── run_experiment.py, scheduler.py, workload.py   # Phase 1 FIFO baseline
│   ├── kv_manager.py                                   # Phase 2 slot pool + KV policies
│   ├── agent_replay.py                                 # Phase 2 trace replay
│   └── react_agent.py                                  # Phase 2 real ReAct agent
├── workloads/         # tracelab_episodes.json, hotpotqa_tasks.json
├── PROJECT_REPORT1027.md
├── README.md
└── requirements.txt
```

Additional Phase 2 scripts and evaluation code implement KV-cost probes, trace replay, KV policies, TTL tuning, and the real-agent evaluation.

---

# Setup

## 1. Create and activate the Python environment

Use Python 3.12 (the pinned `numpy==2.0.2` does not install on 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Install llama.cpp

On macOS:

```bash
brew install llama.cpp
```

On Windows or Linux, download a prebuilt release from <https://github.com/ggml-org/llama.cpp/releases> (we used build b11748 with CUDA 13.4 on Windows).

## 3. Start llama-server

```bash
mkdir -p results/slot_cache
```

```bash
llama-server \
  -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M \
  --host 127.0.0.1 \
  --port 8080 \
  -c 32768 \
  -np 8 \
  -ngl 99 \
  --cache-ram 0 \
  --slot-save-path results/slot_cache
```

Verify:

```bash
curl http://127.0.0.1:8080/health
```

Expected response:

```json
{"status":"ok"}
```

---

# Running the Phase 1 Sweep

```bash
python -m src.run_experiment \
  --concurrency 1 2 4 8 16 32 \
  --trials 3 \
  --seed 42 \
  --tasks-per-type 12 \
  --context-repeats 30 60 120
```

Each sweep creates a new output directory under:

```text
results/runs/
```

---

# Generating Phase 1 Results

```bash
python analysis/analyze.py \
  --results-dir results/runs/<RUN_DIRECTORY>
```

Example outputs include:

```text
summary_aggregated.csv
request_summary.csv

throughput_vs_concurrency.png
completion_time_vs_concurrency.png
queue_time_vs_concurrency.png
execution_time_vs_concurrency.png
tail_latency_vs_concurrency.png
memory_vs_concurrency.png
failure_rate_vs_concurrency.png

report.html
```

---

# Phase 1 Extras: TTFT/TPOT, Idle Periods, KV Costs

```
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

---

# Running Phase 2: Per-Pause KV Policies

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
---

# Results Map

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
---

# Data Sources

- **TraceLab** coding-agent trace dataset, SyFI Lab, University of Washington, CC BY 4.0 (<https://tracelab.cs.washington.edu>). `workloads/tracelab_episodes.json` is a scaled sample; to rebuild it, download `syfi_coding_trace.jsonl.gz` (release v0.0.2) and run `python scripts/prepare_tracelab.py <path>`.
- **HotpotQA**, Yang et al., EMNLP 2018, CC BY-SA 4.0 (<https://hotpotqa.github.io>). `workloads/hotpotqa_tasks.json` is a 60-question sample of the distractor validation split; to rebuild it, run `scripts/prepare_hotpotqa.py` on the Hugging Face parquet file (needs `pyarrow`, which is not a project dependency).

---

# Reproducibility

The project includes several controls added during development:

- fixed-seed trial ordering,
- repeated trials,
- explicit server configuration logging,
- cache clearing between trials,
- request-level timing and token statistics,
- TTFT / TPOT analysis,
- failure and timeout tracking,
- separate output directories,
- development/test split evaluation for policy tuning.

Earlier experiments revealed that reusable prompt caches can significantly distort context-size comparisons, so reported measurements explicitly control reusable state.

---

# Current Status

## Phase 1

- [x] FIFO fixed-concurrency baseline
- [x] mixed workload
- [x] context-length sweep
- [x] concurrency sweep through 32
- [x] multi-step tasks with tool delays
- [x] request-level instrumentation
- [x] latency / throughput / memory metrics
- [x] tail-latency measurements
- [x] cache-reset controls
- [x] reproducible repeated trials
- [x] automatic plots and HTML report

## Phase 2

- [x] KV cost characterization
- [x] TTFT / TPOT analysis
- [x] idle-period sweep
- [x] scaled TraceLab agent workload
- [x] per-pause agent trace replay
- [x] on-demand min-waste policy
- [x] TTL tuning
- [x] held-out test evaluation
- [x] real ReAct agent on HotpotQA
- [x] tail-aware min-waste reclaim policy
- [x] confidence intervals, boundary study, and larger-context real-agent runs
- [ ] final validation on AWS
- [ ] final ablations and cleanup
- [ ] final report / presentation
