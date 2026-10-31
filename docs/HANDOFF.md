# Handoff: where the project stands and how to continue

Written 2026-10-30 night by the Claude session that did the work, for the next session / the team.
Read `README.md` for the task and results, `AGENTS.md` for the rules. This file is the *working* state.

## 1. The goal in one paragraph
Soup Kernel Challenge (Decentrathon 6.0): make the fixed `soup train` run (Qwen2.5-1.5B, 4-bit NF4, LoRA r=16
on q/v, batch 4, <=512 tokens, 64 steps, fp16) faster and/or lighter on a Tesla T4 with our OWN `@triton.jit`
kernels, without changing the task. Score = 20 (all 10 checks pass) + 35*ln(speedup)/ln(best) + 30*ln(mem ratio)/ln(best)
+ 15 README. Deadline: 2026-10-31 23:59 Astana (UTC+5). Only the platform-created team repo is checked:
copy this repo's final state into it (`git push <platform-repo> main`).

## 2. State of `main` (code frozen at the measured commit)
Submission result (Kaggle T4, saved version 1, full harness, 10/10 checks): base 87.77 s / 14,345 MiB
-> ours 62.10 s / 5,441 MiB (x1.413, 2.636x less memory), eval loss 2.04964 -> 2.04952,
our kernels 10.8% of training time. The earlier Colab pair measured 76.4 -> 63.5 s; its separate
two-run seed repeat measured 80.8 -> 66.2 s. See README and SUBMISSION for the evidence and limits.
What is in `solution/`: chunked head+cross-entropy (ce_loss.py, head_patch.py), RMSNorm (rmsnorm.py), RoPE (rope.py),
NF4 dequant for bitsandbytes fwd+bwd (nf4_dequant.py), fused residual-add+RMSNorm (add_rmsnorm.py),
packed layers = no padding rows + fused layer boundaries (layer_tail.py), tied-embedding fp16 compaction (embed16.py),
optional MLP recompute (recompute.py, OFF: +16% time for -27% memory, a bad trade), amp.py helper.

## 3. Experiment branches (all evaluated; none is a pending submission candidate)
| Branch | Idea | Final result |
|---|---|---|
| `m2/embed16-safe` | Safer tied-embedding dtype handling | Four embed16 tests passed on T4; optional robustness change, not needed for the measured submission. |
| `q/varlen-attn` | Per-sample causal SDPA on packed rows | T4 A/B showed no speed or memory gain; rejected. |
| `s/startup` | Reduce startup and Triton compile time | No material submission gain; rejected. |
| `c2/fused4bit` | Fuse NF4 decode into a Triton GEMM | 10–200x slower than dequant plus cuBLAS on tested shapes; rejected. |
Other tested ideas and evidence are recorded in sections 9–11 and in README. No further GPU experiments are planned for this submission.

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

## 6. Remaining work (updated 2026-10-31)

The Kaggle run, team details, code snapshot, Pitch deck, and 2:56 demo video are recorded in
`docs/CHECKLIST_STATUS.md` and `SUBMISSION.md`. The only unresolved handoff is to authenticate on
HackAlem as the captain, confirm jury access to the private platform repository, and finish the
platform submission before the deadline. The current `4spanch1k` GitHub session cannot open that
private repository. Never download local AI models (user rule).

## 7. Remaining steps
1. In an anonymous browser session, open the public Kaggle, Pitch, and video links from `SUBMISSION.md`.
2. The captain signs in to HackAlem, confirms the team submission, and verifies that judges can access the platform repository and recorded snapshot.
3. Keep `solution/` identical to the measured code; do not resume rejected experiments for this submission.

## 8. Codex takeover prompt (paste this into Codex if Claude's limit runs out)

```text
You are taking over final delivery for the Soup Kernel Challenge. Repository: https://github.com/4spanch1k/soup-kernel-challenge
(local clone: /Users/aspanch1k/decentrathon-6.0/soup-kernel-challenge). Read AGENTS.md, README.md,
SUBMISSION.md, docs/CHECKLIST_STATUS.md, and docs/HANDOFF.md. The measured Kaggle T4 submission is
87.77 -> 62.10 s, 14,345 -> 5,441 MiB, eval loss 2.04964 -> 2.04952, with all 10 checks passed;
the separate Colab results are labeled as Colab in the documents.

The code is frozen. Do not edit solution/, harness/, tests/, or task/, and do not run or revive
rejected GPU experiments. Do not download local AI models. The only outstanding work is external:
the team captain must authenticate on HackAlem, confirm the submission and judge access to the
private platform repository, then verify the public links in an anonymous browser session. Do not
claim platform submission or judge access without seeing confirmation. Never request or handle
another person's password or verification code.

Deadline: 2026-10-31 23:59 Astana time.
```

## 9. Experiment log (T4, Colab, 2026-10-31 early morning)
| Branch | Result | Verdict |
|---|---|---|
| `q/varlen-attn` (per-sample causal SDPA on packed rows) | tests 5/5 pass after fixing `attention.num_heads` (commit 401fc2f). A-B-A-B solution-only wall: main 61.8 / 68.0 s, varlen 66.9 / 68.3 s; peak identical 5,441 MiB | NO gain (slightly slower, 3 SDPA launches per layer + repeat_kv); do not merge |
| `c2/fused4bit` (NF4 decode fused into a Triton GEMM) | kernel matches reference (12 tests pass; one wrapper test fails on a test-side AttributeError) but is 10-200x SLOWER than dequant+cuBLAS on every shape (scripts/bench_gemm4bit.py) | rejected; cuBLAS wins on sm75 as predicted |
| `s/startup` | scripts/startup_breakdown.py (run with `PYTHONPATH=. python scripts/startup_breakdown.py`): cold Triton compile of ALL our kernels = 4.7 s with an empty cache (first call 1.9 s incl. Triton init; others 0.1-0.5 s each); no recompiles across T=200/311/512. Import torch 1.5 s, transformers+trl+peft+bnb 8.2 s (same for base), apply() 1.1 s | the ~4.7 s compile is the only startup slice we own; its RoPE/embed16 tweaks are marginal, not merged |
| MLP recompute | +16% time for -27% memory | off |
| `m2/embed16-safe` | 4 embed16 tests pass on T4; the PEFT lifecycle test needs a peft/torchao-compatible env (skips otherwise) | harmless robustness; merge optional |
Also measured: wall vs loop: base 76.4 s wall / 54.9 s loop; ours 63.5 / 42.1 (loop includes ~5 s of first-step Triton compile).
Open idea (rule-ambiguous, needs the user's OK): warm up our kernels in a Python THREAD inside apply() while Soup loads the model (compile is CPU work), hiding ~4.7 s of the wall time. Not a background process and not a cache, but a jury could read it as "background work": decide with the user and document it in README if used.

## 10. Robustness check on main (2026-10-31 morning)
`python harness/run_task.py --repeats 2 --seed 2026 --out runs_seed2026 && python harness/check_rules.py --result runs_seed2026/result.json` on main (29184d9), Colab T4:
ALL 10 CHECKS PASS. base 80.8 s / 14,339 MiB, ours 66.2 s / 5,419 MiB (x1.221, 2.646x less), eval loss +0.016%, kernels 8.8% (lowest run),
loss-curve gap <= 0.05%. `w/load-phase` (Codex): no value-preserving saving > 0.5 s in the model-load phase (analysis in that branch,
docs/LOAD_PHASE_ANALYSIS.md, scripts/load_phase_timeline.py). Decision: the thread-warm-up idea is NOT used (rule risk).
Remaining ideas by value: LoRA-path fusion (-150..300 MiB, -5..15 ms), nothing else big is known. main is the submission candidate.

## 11. `l/lora-fuse` (Codex, tested on T4): rejected
Tests: 2 pass, the PEFT-dependent test skips/fails on Colab because of an old preinstalled torchao (env issue). With SOUP_LORA_FUSE=1 the
harness reports `_add_scaled_kernel, _dropout_backward_kernel, _dropout_cast_kernel` as NOT launched: the patch never fired because for
bitsandbytes 4-bit layers PEFT uses `peft.tuners.lora.bnb.Linear4bit` (its own forward), not the generic `lora.layer.Linear` that was patched.
Result with the (inactive) patch: 63.2 s / 5,441 MiB = identical to main. Microbenchmark of the fused call alone (scripts/bench_lora_fuse.py):
slower at N=826 (1.26 vs 0.94 ms per q call, ~+17 ms/step over 56 calls) and only ~-12 MiB retained per layer at N=2048 (~-330 MiB at the peak batch),
so even if re-targeted the net score effect is about zero or negative. Do not pursue unless a clearly cheaper design appears.
Conclusion: the measured code is frozen and all experiment branches are rejected. The only remaining handoff is platform access and submission (see section 6).
