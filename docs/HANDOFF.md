# Handoff: where the project stands and how to continue

Written 2026-10-31 night by the Claude session that did the work, for the next session / the team.
Read `README.md` for the task and results, `AGENTS.md` for the rules. This file is the *working* state.

## 1. The goal in one paragraph
Soup Kernel Challenge (Decentrathon 6.0): make the fixed `soup train` run (Qwen2.5-1.5B, 4-bit NF4, LoRA r=16
on q/v, batch 4, <=512 tokens, 64 steps, fp16) faster and/or lighter on a Tesla T4 with our OWN `@triton.jit`
kernels, without changing the task. Score = 20 (all 10 checks pass) + 35*ln(speedup)/ln(best) + 30*ln(mem ratio)/ln(best)
+ 15 README. Deadline: 2026-10-31 23:59 Astana (UTC+5). Only the platform-created team repo is checked:
copy this repo's final state into it (`git push <platform-repo> main`).

## 2. State of `main` (code frozen at the T4-measured commit)
Latest measured (Colab T4, full harness, 10/10 checks): base 76.4 s / 14,345 MiB -> ours 63.5 s / 5,441 MiB
(x1.20, 2.64x less memory), eval loss -0.006%, our kernels 9.6% of training time. See README "Results".
What is in `solution/`: chunked head+cross-entropy (ce_loss.py, head_patch.py), RMSNorm (rmsnorm.py), RoPE (rope.py),
NF4 dequant for bitsandbytes fwd+bwd (nf4_dequant.py), fused residual-add+RMSNorm (add_rmsnorm.py),
packed layers = no padding rows + fused layer boundaries (layer_tail.py), tied-embedding fp16 compaction (embed16.py),
optional MLP recompute (recompute.py, OFF: +16% time for -27% memory, a bad trade), amp.py helper.

## 3. Open experiment branches (each from main; none is merged; each needs a T4 A/B before merging)
| Branch (worktree) | Idea | Expected | Status when this was written |
|---|---|---|---|
| `m2/embed16-safe` (soup-m) | embed16 dtype-safe head outside autocast + PEFT-order lifecycle test | robustness only | pushed, not yet run on T4 |
| `q/varlen-attn` (soup-q) | Codex: per-sample causal SDPA on packed rows, no scatter/gather | -5..8% time | Codex was implementing |
| `s/startup` (soup-s) | Codex: ~28 s of the 63 s wall are OUTSIDE the training loop (imports, model load, Triton JIT with empty cache); cut Triton recompiles; adds scripts/startup_breakdown.py | up to several seconds | Codex was implementing |
| `c2/fused4bit` (soup-f) | Codex bake-off: fused NF4 GEMM in Triton vs our dequant+cuBLAS; default OFF, microbench first (scripts/bench_gemm4bit.py) | long shot (cuBLAS is strong on T4) | Codex was implementing |
Other ideas not built: fuse the LoRA path (dropout+A+B+add; -5..15 ms, -150..300 MiB), Triton flash-attention for sm75.

## 4. Decision rule for merging an experiment
Run `bash scripts/kaggle_3_full.sh` for main and for the branch IN THE SAME Colab SESSION (Colab speed drifts: base
was 75.9-80.9 s across sessions; always compare speedup against the base of the same run). Merge only if all 10 checks
pass AND (speedup improves by >= 3% relative, or peak memory drops >= 5%) AND check 4 stays >= 6% (margin).
A speed gain under 5% counts as a tie in the score, so tiny gains are worth little. Repeat a promising run twice.

## 5. How to run things (no local GPU on the Mac)
* Colab T4 in the app's built-in browser (the user is signed in to Google there; Kaggle has no GPU until the user
  verifies the phone). Resize the tab first: `resize_window 1400x900` (it resets every turn), click the cell,
  `cmd+a`, type ONE command, `cmd+Return`. Pitfalls: do not `pkill -f` a string that is in your own command;
  `!` cells can ignore interrupt for a while; long output is buffered by `| tail`.
* Standard cell: `!cd repo && git fetch -q origin && git checkout -q <branch> && python -m pytest solution -q -x -p no:cacheprovider 2>&1 | tail -25 && bash scripts/kaggle_3_full.sh 2>&1 | tail -40`
  (~13 min). If the runtime restarted: `!git clone https://github.com/4spanch1k/soup-kernel-challenge.git repo && cd repo && bash scripts/kaggle_1_tests.sh`.
* Other scripts: `scripts/kaggle_2_quick.sh` (solution only + profile), `scripts/top_kernels.py` (versions + top kernels),
  `scripts/profile_shapes.sh` (ops with shapes), `scripts/pad_fraction.py` (43% of batch rows are padding).
* Codex CLI (not on PATH): `~/.vscode/extensions/openai.chatgpt-*/bin/macos-aarch64/codex exec -s workspace-write -C <worktree> --skip-git-repo-check -o out.md - < prompt.md`
  run in the background; its sandbox cannot write the shared git index (commit for it) and cannot reach the network.

## 6. Things only the human can do
1. Merge approvals (an automated check blocks merging without the user's explicit chat approval).
2. Verify the phone in Kaggle (kaggle.com/settings) and run `docs/RUN_ON_KAGGLE.md` there (`COMMIT` = final sha).
3. Record the <=3 min video (`docs/DEMO_VIDEO_SCRIPT.md`), finish the presentation (draft deck artifact, outline in `docs/PRESENTATION_OUTLINE.md`).
4. Fill team name/members/links/final sha in `SUBMISSION.md`; copy the final code into the platform repository.
Never download local AI models (user rule).

## 7. Next steps in order of expected value
1. T4 A/B of `s/startup` (largest untouched slice), then `q/varlen-attn`; merge winners per the rule above.
2. Seed check of the final code: `python harness/run_task.py --repeats 2 --seed 2026 --out runs_seed2026 && python harness/check_rules.py --result runs_seed2026/result.json`.
3. Edge batch test with 512-token rows (script in the Codex review notes: Python snippet that writes /tmp/soup_edge512.jsonl, then `--only solution --train-file /tmp/soup_edge512.jsonl`).
4. Refresh README/SUBMISSION numbers if main changes; keep `solution/` identical to what was measured.
