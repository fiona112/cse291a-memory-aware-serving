# KV cost characterization (Phase 1 addition)

Measured Oct 25, 2026 on one laptop: NVIDIA RTX 5070 Laptop GPU (8 GB), llama.cpp build b11748 (CUDA 13.4), Qwen2.5-1.5B-Instruct Q4_K_M, `-c 32768 -np 4 -ngl 99 --cache-ram 0 --slot-save-path <dir>`. Script: `experiments/kv_cost_probe.py`. Raw data: `kv_cost_*.csv` in this folder.

**Bottom line:** on this GPU, saving a context to disk and restoring it costs about 4–5x less than recomputing it (at 4k tokens: 52 ms restore + 75 ms save vs. 573 ms recompute). Offloading is therefore a real option for the Phase 2 policy, not just a theoretical one.

## Feasibility checks

| Capability | Result |
| --- | --- |
| Pin a request to a slot (`"id_slot": N`) | Works; the response confirms the slot |
| Save a slot to disk (`POST /slots/N?action=save`) | Works; reports tokens, bytes, `save_ms` |
| Restore into an erased slot (`action=restore`) | Works; the next request reuses all restored tokens (only the new suffix is prefilled) |
| CPU-RAM prompt cache (`--cache-ram 8192`) | Works; a context pushed out of its slot came back with 100% of its tokens reused |

## KV size per token

| KV type | Formula | Measured (saved slot file) |
| --- | --- | --- |
| f16 | 2 × 28 layers × 2 KV heads × 128 dims × 2 bytes = 28,672 B | 28.2 KiB/token (112.8 MiB for 4,096 tokens) |
| q8_0 | 2 × 28 × 2 × 128 × 1.0625 B ≈ 15,232 B | 15.0 KiB/token (59.9 MiB for 4,096 tokens) |

VRAM used by the server process (whole-GPU, minus 1,151 MiB idle): **2,034 MiB with f16 KV vs. 1,621 MiB with q8_0** for a 32,768-token budget. The 413 MiB saving matches the formula (~420 MiB).

## Cost of each option vs. context length (f16, flash attention on, median of 3)

| Context (tokens) | Recompute (ms) | Preserve: suffix prefill only (ms) | Save to disk (ms) | Restore from disk (ms) | File size (MiB) |
| --- | --- | --- | --- | --- | --- |
| 256 | 37 | 12 | 10 | 6 | 7.7 |
| 512 | 62 | 13 | 13 | 7 | 14.3 |
| 1,024 | 128 | 14 | 21 | 13 | 28.5 |
| 2,048 | 251 | 14 | 41 | 28 | 56.4 |
| 4,096 | 573 | 15 | 75 | 52 | 112.8 |
| 6,144 | 894 | 15 | 106 | 77 | 168.5 |

Per token: recompute ≈ 0.14 ms, save ≈ 0.018 ms, restore ≈ 0.013 ms. Preserving costs almost no time, but it holds ~28 KiB of KV per token in GPU memory for the whole tool call.

**CPU-RAM cache (`--cache-ram 8192`), wall time to get a context back and answer:** 42 ms (1k tokens), 67 ms (2k), 94 ms (4k), vs. 84 / 160 / 337 ms to recompute in the same server run. The server does not report the RAM-load time separately, so this is wall time including one decode step and HTTP overhead.

## KV quantization (q8_0)

- Halves KV memory (above), so the same budget holds about 2x the contexts.
- Save/restore files are half the size, and restore is ~30% faster (4k tokens: 37 ms vs. 52 ms).
- Decode was ~6% slower in one spot check (6.01 vs. 5.68 ms/token). Output quality is not yet checked.

## TTFT and TPOT from the existing team runs

Computed by `analysis/ttft_tpot.py` from the committed request logs (all workloads and context sizes pooled). TTFT is server-side (prefill + one decode step); "server wait" is time queued inside llama-server.

| Concurrency | Hyeonung run: TTFT p50 / p95 (ms) | TPOT p50 (ms) | Server wait p50 (ms) | Fiona run: TTFT p50 / p95 (ms) | TPOT p50 (ms) | Server wait p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 / 5,412 | 18.5 | 4 | 50 / 3,145 | 10.2 | 3 |
| 2 | 108 / 5,569 | 37.8 | 18 | 69 / 3,166 | 20.4 | 8 |
| 4 | 184 / 5,663 | 58.6 | 29 | 106 / 3,210 | 30.8 | 16 |
| 8 | 246 / 5,880 | 52.7 | 35 | 155 / 3,222 | 32.4 | 21 |
| 16 | 240 / 5,971 | 43.2 | 6,757 | 145 / 3,204 | 26.8 | 3,820 |
| 32 | 237 / 5,987 | 42.2 | 16,580 | 147 / 3,295 | 23.9 | 9,427 |

- **p95 TTFT is dominated by uncached long-context prefill:** about 5.4 s (Hyeonung) and 3.1 s (Fiona) for a ~3,900-token prompt, vs. ~0.57 s on the RTX 5070. Recompute is 5–10x more expensive on those machines, so avoiding recompute matters even more there.
- **Above 8 concurrent requests, the extra requests just wait inside llama-server** (median wait jumps from ~35 ms at c=8 to 6.8 s at c=16), because the server has only 8 slots. Raising client concurrency past `-np` does not add parallelism.

## Caveats

- **Disk restore was likely served from the OS file cache,** since each file was read right after it was written. A true cold SSD read will be slower. On the Linux AWS machine we can drop the page cache (`echo 3 > /proc/sys/vm/drop_caches`) to measure it.
- **Recompute times vary between server restarts** (4k tokens: 337–591 ms across runs). Compare options within the same server run, as the table above does.
- Laptop numbers only; the AWS GPU will differ.

## What this means for Phase 2

1. All three options are implementable with llama.cpp's existing API, so the per-pause policy is feasible.
2. Offload (disk or CPU RAM) is several times cheaper than recompute at every length tested, and the gap grows with context length.
3. Preserve is the fastest when the tool returns quickly, but it ties up ~28 KiB/token of a fixed KV budget, which is the trade-off min-waste has to weigh.
4. q8_0 KV is a cheap way to double capacity and can be a secondary knob.
