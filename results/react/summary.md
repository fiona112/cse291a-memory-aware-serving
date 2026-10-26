# Real ReAct agent on HotpotQA (M5)

Same setup as `results/replay` (RTX 5070 laptop, 4 slots × 4,096 tokens, Qwen2.5-1.5B-Instruct Q4_K_M). Agent: `python -m src.react_agent`; tasks: `workloads/hotpotqa_tasks.json` (60 HotpotQA distractor questions, CC BY-SA 4.0; 30 dev, 30 test). The model writes its own `Thought` / `Action: search[...]` / `Action: finish[...]` steps, search really ranks each question's 10 Wikipedia paragraphs (top 2 returned), and each tool call also waits a delay sampled from the real TraceLab tool times (the same delay for a given question and step under every policy). Test split, 3 trials per cell.

**Bottom line:** for this real agent, the KV policy barely matters for latency, because its contexts are small (about 900 tokens on average, 2.9 LLM calls per question), so recomputing them is cheap. Offload still cut prefill work by up to 31% at the highest load without hurting latency or answers. Keeping slots reserved (preserve) or capping admission (K = 6) made things worse. This is one of the "little benefit" cases the plan called for.

## Latency and work (test split)

| Policy | p50 / p95 completion (s), rate 1 | rate 2 | rate 4 | Prefill tokens computed per trial (rate 4) |
| --- | --- | --- | --- | --- |
| release (default) | **1.6** / **17.8** | 2.1 / **18.2** | **3.0** / **18.9** | 16,587 |
| release + K=6 | 1.7 / 19.0 | 6.1 / 24.5 | 8.1 / 24.1 | 12,607 |
| preserve | 9.4 / 24.3 | 15.4 / 28.5 | 17.5 / 29.8 | 13,266 |
| offload | 1.7 / 18.7 | 2.1 / 18.9 | 3.6 / 19.2 | **11,361** |
| ttl 1 s | 1.7 / 20.7 | 2.4 / 20.0 | 5.8 / 22.9 | 15,265 |
| min_waste | 2.8 / 19.1 | 4.9 / 21.0 | 3.8 / 23.5 | 13,348 |
| min_waste + ready | 2.9 / 20.2 | 5.9 / 22.3 | 5.1 / 21.6 | 12,682 |
| min_waste + tail + ready | — | **2.0** / 18.9 | 3.5 / 19.2 | 12,034 |

The tail-aware variant (see `results/replay/summary.md`) was run afterwards at rates 2 and 4 only: it was the fastest policy at rate 2 (mean non-tool time 1.33 s vs. 1.39 s for release) and close to release at rate 4.

At rate 4, release recomputed 34 of 40 returning steps per trial on average, while offload restored 35 from disk. With contexts this small, a recompute costs about as much as a restore, so both give similar latency.

## Answer quality (validity check)

| | Value |
| --- | --- |
| Exact match, all policies and rates | 0.14–0.19 |
| F1, all policies and rates | 0.20–0.26 |
| Agreement of each policy's answers with release's answers (same question, same trial) | 64–79% |
| Agreement of a policy with **itself** across its own trials | 60–80% (mean 72%) |

Answers differ between runs even at temperature 0, because batched GPU arithmetic depends on which requests are processed together. Cross-policy agreement is in the same range as a policy's agreement with itself, and accuracy does not drop under any policy. So the KV policies do not change answer quality beyond normal run-to-run variation. Per-trial answers are in each run's `trial_*_answers.csv`; the agreement table is `answer_agreement.csv` (`python -m analysis.answer_agreement results/react/test`).

## Notes

- The 1.5B model is a weak HotpotQA agent (EM ≈ 0.17). That is fine for a systems comparison, but the low accuracy limits what the quality check can detect.
- Search returning the top 2 paragraphs (up to 150 words each) instead of 1 raised exact match from 0.03 to about 0.17 in a smoke test, and made contexts closer to real tool outputs.
- Memory pressure only appears at high load (≥ 2 questions per second with 4 slots); at rate 0.6 no agent ever waited for a slot.
