# Final note for the next agent (Codex or anyone), written 2026-10-30 night when Claude's limit ran low

Read first: AGENTS.md, docs/HANDOFF.md, docs/FINISH_CHECKLIST.md, docs/CODEX_VS_CLAUDE.md. Deadline: 2026-10-31 23:59 Astana (UTC+5).

## State
* `main` = 6f3a11e (frozen, T4-measured: 63.5 s / 5,441 MiB vs base 76.4 s / 14,345 MiB, 10/10 checks). Never change `solution/` on main without a full T4 run (`bash scripts/kaggle_3_full.sh`), and never merge without the user's explicit "влей PR #N" in chat.
* The user's teammate `yujio2x` (write access) finishes the human tasks in docs/FINISH_CHECKLIST.md: Kaggle notebook, team data, video, presentation, push to the platform repo.
* Two experiment branches exist (neither merged):
  * `x/measure`: scripts/t4_*.py, scripts/t4_measure.sh and docs/T4_MEASUREMENTS.md (measured on Colab T4: Triton 3.6 `tl.dot` uses NO tensor cores on sm75, so no own Triton GEMM/attention; attention already runs the cutlass mem-efficient kernel, 51.7 ms of the 454.6 ms step; peak memory is almost all torch tensors).
  * `x/causal-attn` (commit 8ee65f0): `solution/layer_tail.py` calls SDPA with `is_causal=True` and expanded KV heads instead of a 4D mask (right padding or no padding), plus tests in solution/test_packed_layers.py. Expected about -1 s (microbenchmark: 1.94 vs 2.94 ms per layer at T=512). It also avoids the math-kernel fallback that a batch without padding triggers (transformers passes `enable_gqa=True` with no mask: 8.4 ms vs 2.9 ms per layer on this T4).

## A run was in progress
A Colab cell was running `python -m pytest solution -q -x` on `origin/x/causal-attn` and then `bash /tmp/t4_ab.sh`: solution-only paired runs of `origin/main` (A) and `origin/x/causal-attn` (B), A-B-A-B on the public rows, plus an edge file where EVERY row is longer than 512 tokens (built only from public rows by scripts/make_edge512.py). Logs: /tmp/ab_<tag>.log and /content/repo/runs_ab/<tag>/ on the Colab runtime; the cell prints `### tag (ref)` and `=> wall .. s peak .. MiB eval loss ..` per run. Colab URL: https://colab.research.google.com/drive/1fEphIzcz3oFL5aRsEHQdWDqCC_nDATnS (the runtime may have been recycled; then redo: clone the repo, `bash scripts/kaggle_1_tests.sh`, then `cp scripts/t4_ab.sh /tmp/ && bash /tmp/t4_ab.sh`, with `git checkout origin/x/causal-attn` first).

## Decision rule (the project's merge rule, section 4 of docs/HANDOFF.md, corrected)
Merge `x/causal-attn` into main only if ALL hold: the solution tests pass on the T4 (including the new causal tests); `bash scripts/kaggle_3_full.sh` on that branch gives 10/10 checks and "79 passed"; check 4 stays >= 6%; peak memory not worse than 5,441 MiB; the edge file (all rows > 512 tokens) runs without error and is not slower than main on it; and in the same session B beats A in both A-B pairs on wall time (expected about 1 s, i.e. about 1.5%; Colab noise is larger, so repeat). If any condition fails or time is short: leave main frozen. Then refresh README/SUBMISSION numbers, re-run Kaggle with the final commit, and re-push the platform repo.
Note on the tie rule: the 5% floor in harness/score.py is applied to each team's own ratio against the base, NOT between our versions; our ratio is already about 1.20, so a 1.5% gain still moves the score slightly (about 0.3-1 point).

## Not allowed / rejected (do not retry)
Triton GEMM or attention on this T4 (no tensor cores), persistent or shipped Triton caches, warm-up threads, allocator settings other than the harness's `expandable_segments:True`, MLP recompute, caching dequantized weights, LoRA fusion, fused NF4 GEMM, varlen attention. Never download local AI models. Do not read dataset rows outside the public ranges 0-255 and 40000-40255.

## Codex takeover prompt
```text
You take over a hackathon repo (https://github.com/4spanch1k/soup-kernel-challenge, local /Users/aspanch1k/decentrathon-6.0/soup-s). Read docs/HANDOFF_FINAL_NOTE.md first, then AGENTS.md, docs/HANDOFF.md, docs/FINISH_CHECKLIST.md. Goal: finish safely before 2026-10-31 23:59 Astana. 1) Check the result of the A/B run described in the note (ask the user to paste the Colab cell output if you cannot see it) and apply the decision rule for branch x/causal-attn. 2) Otherwise keep main frozen. 3) Help the teammate yujio2x with docs/FINISH_CHECKLIST.md: fill README/SUBMISSION TODOs only with real facts, never invent links. Never merge without the user's explicit approval in chat, never edit harness/, tests/, task/, never download local models.
```

## RESULT of the A/B run (2026-10-31, Colab T4, torch 2.11.0, triton 3.6.0; solution-only runs, MEASURED)
| Run | main (6f3a11e) | x/causal-attn (72a5f19) |
|---|---|---|
| public rows, pair 1 | 59.93 s, 5,441 MiB, eval 2.04949 | 59.87 s, 5,441 MiB, eval 2.04949 |
| all-512-token edge file (no padding in any batch) | 87.1 s, 6,961 MiB, eval 2.11128 | 88.95 s, 6,961 MiB, eval 2.11128 |
Solution tests: 52 passed (including the new dense/no-mask/left-padding tests). The second public pair was still running when this was written.
Verdict: the causal branch gives NO measurable gain (-0.06 s on the public rows, +1.8 s on the edge file, same memory, same loss). The microbenchmark gain
(1.94 vs 2.94 ms per layer at T=512) did not show up end to end. By the decision rule: do NOT merge; `main` stays frozen. Also: main handles an all-512-token
edge batch correctly (87.1 s, 6,961 MiB, no failure, same peak as the branch), so the feared math-attention fallback did not change memory here.
