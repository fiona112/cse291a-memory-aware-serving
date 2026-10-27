# Per-pause KV policies on TraceLab agent replay (M3–M4)

Laptop RTX 5070 (8 GB), llama.cpp b11748, Qwen2.5-1.5B-Instruct Q4_K_M, **4 slots × 4,096 tokens** (`-c 16384 -np 4 --cache-ram 0 --slot-save-path`). Workload: `workloads/tracelab_episodes.json` (40 scaled TraceLab episodes; 20 dev, 20 test). Runner: `python -m src.agent_replay`. Analysis: `python analysis/compare_policies.py results/replay/<group>`. Every run produced exactly the target output lengths (`output_mismatches = 0`), so no policy gains by generating less.

**Bottom line:** under memory pressure, keeping each agent's KV across tool calls instead of letting it be overwritten avoids ~86% of the prefill work. Plain **offload at every tool call** was the most robust fixed policy. The final **tail-aware min-waste with ready-step priority** (selected on dev, confirmed on test) beat it, cutting scheduler-added (non-tool) time by 13–17% vs. offload and 31–33% vs. the default, with 44–63% fewer disk writes. Fixed parameters tuned on dev (admission limit K = 6, TTL = 1 s) did not carry over to the test split, whose tool times are much longer.

## Policies

| Name | At each tool call |
| --- | --- |
| release | Free the slot; KV survives only if nobody reuses the slot (llama-server default behavior) |
| preserve | Keep the slot reserved until the agent returns |
| offload | Save the slot's KV to disk, free the slot, restore it on return |
| ttl (Continuum-style) | Keep the slot for a fixed TTL, then release it |
| min_waste (InferCept-style, on demand) | Keep the KV; when someone needs a slot, offload the idle agent with the longest predicted remaining tool time |
| `_ready` | Agents returning from a tool get slots before new episodes (with aging) |
| `_k6` | Admit at most 6 episodes at once (the best fixed setting on dev) |

## Test split (5 trials per cell)

| Policy | Rate 0.3: p50 / p95 completion (s) | Mean non-tool time (s) | Rate 0.6: p50 / p95 completion (s) | Mean non-tool time (s) | Prefill tokens computed (rate 0.6) |
| --- | --- | --- | --- | --- | --- |
| release (default) | 32.9 / 144.2 | 14.3 | 42.7 / 145.8 | 22.1 | 280,106 |
| release + K=6 (tuned on dev) | 48.3 / 147.7 | 27.4 | 59.5 / 164.0 | 34.5 | 74,531 |
| preserve | 90.4 / 199.9 | 59.7 | 101.0 / 221.8 | 69.9 | 38,250 |
| **offload** | **27.6** / 142.8 | 11.6 | **35.7** / **143.2** | **17.8** | 38,250 |
| ttl 1 s (Continuum) | 38.3 / 146.1 | 14.6 | 47.5 / 149.6 | 21.5 | 119,333 |
| min_waste | 29.3 / 141.9 | 11.5 | 41.4 / 144.9 | 21.3 | 38,251 |
| min_waste + ready | 28.5 / 141.0 | **11.0** | 44.7 / 145.6 | 24.3 | 38,753 |
| min_waste + K=6 | 48.0 / 147.2 | 26.4 | 60.1 / 163.4 | 34.8 | 38,251 |
| min_waste + ready + K=6 | 46.9 / 147.0 | 25.9 | 60.6 / 164.0 | 35.0 | 38,402 |

p95 completion is dominated by the episodes' own tool time (up to ~130 s of real tool waits), so **mean non-tool time** (completion minus tool time) is the cleaner measure of what the scheduler adds.

## Findings

1. **The default behavior wastes most of the prefill work.** At rate 0.6, release recomputed 70% of returning agents' contexts (147 of 211 returns) and computed 7.3x more prompt tokens than any KV-keeping policy.
2. **Preserve fails when tools are long.** Reserved slots sit idle during 30 s tool calls while others queue (mean slot wait 5.4 s vs. 0.65 s for offload).
3. **Offload wins because disk save/restore is cheap here** (~67 ms per save at ~3k tokens) compared with both recompute and idling a slot. With a slower disk or a faster recompute, the trade-off could flip.
4. **Admission limits tuned on dev did not generalize.** K = 6 was best on dev (p50 20 s vs. 28 s at rate 0.3) but worst on test, because test episodes hold their admission slot through long tool calls. A fixed K is sensitive to the tool-time distribution.
5. **On-demand min-waste trailed plain offload at high load.** Its tool-time predictor (moving average, median ≈ 0.3 s) underestimates the long tail, so idle agents in long tools were not reclaimed promptly (mean slot wait 1.17 s vs. 0.65 s).
6. **Ready-step priority helped slightly at rate 0.3 and hurt slightly at 0.6.**

## Tail-aware min-waste (selected on dev, confirmed once on test)

Finding 5 pointed to the predictor: TraceLab tool times are heavy-tailed, so a call that has already run long tends to keep running. With `--tail-aware`, once an idle agent has waited past its predicted tool time, its remaining time is estimated as the time it has already waited, and reclaim decisions are re-checked every 0.25 s while agents are waiting. The variant was chosen on the dev split (`results/replay/tail_dev`: min_waste + tail + ready beat offload by 19–24% in p50) and then run once on the test split with offload as a same-session control (`results/replay/tail_test`, 5 trials).

| Test split, same session | Rate 0.3: p50 / p95 (s) | Mean non-tool (s) | Rate 0.6: p50 / p95 (s) | Mean non-tool (s) | Disk saves per trial |
| --- | --- | --- | --- | --- | --- |
| offload | 27.8 / 142.6 | 11.6 | 35.2 / 143.2 | 17.5 | 211 |
| min_waste + tail | 26.2 / 141.2 | 9.8 | 33.6 / 142.3 | 15.4 | 79 / 119 |
| **min_waste + tail + ready** | 26.5 / 141.3 | **9.6** | **31.5** / 142.6 | **15.2** | 78 / 118 |

- **Best policy on the test split at both rates.** Against offload, non-tool time drops 17% at rate 0.3 and 13% at rate 0.6. Against the default release (M4 session), non-tool time drops 33% (14.3 → 9.6 s) and 31% (22.1 → 15.2 s), and p50 drops 19% (32.9 → 26.5 s) and 26% (42.7 → 31.5 s).
- **It needs far fewer disk writes:** it saves KV only when a slot is actually needed (44–63% fewer saves than offload).
- The fix was motivated by a mechanism (heavy-tailed tool times), selected on dev, and run on test once; the earlier `min_waste` results stay in this file for comparison.
- On the real ReAct agent (`results/react/summary.md`) it was the fastest policy at 2 questions/s and close to release at 4 questions/s, as expected for small contexts.

## Cases where KV policy should help little (test split, 3 trials)

| Case | release p50 / p95 (s) | Best alternative p50 / p95 (s) | Takeaway |
| --- | --- | --- | --- |
| Low load (rate 0.15) | **26.8** / 142.4 | ttl 1 s: 28.6 / 143.1 | Little pressure, so little to gain; K = 6 only adds queueing |
| No memory pressure (8 slots, rate 0.6) | **38.9** / 143.8 | ttl 1 s: 42.5 / 144.2 | Without slot contention, KV policy has nothing to fix |
| Short tools (tool time × 0.1, rate 0.6) | 45.3 / 56.1 | ttl 1 s: **17.1** / 41.6; min_waste + ready + K=6: 19.9 / **32.2** | With short tools, contention is constant and recompute dominates, so keeping KV helps the most |

## Caveats

- One laptop GPU; recompute is cheap here (~0.14 ms/token). On slower machines (teammates' runs showed 5–10x slower prefill) avoiding recompute should matter more.
- Dev and test splits are only 20 episodes each, and tool times are heavy-tailed, so the splits differ a lot. Conclusions about tuned parameters (K, TTL) are fragile.
- Disk save/restore was measured warm (OS file cache); a cold SSD read will be slower.
