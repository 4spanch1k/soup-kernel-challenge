# Submission: mononyx

## Links

- Repository: https://github.com/BAITC-Hacks/hack-b6e5a4cf-mononyx
- Submission code commit: https://github.com/BAITC-Hacks/hack-b6e5a4cf-mononyx/commit/a412248dcf19b1bbfd13ebf7f93112eb0322abe2. The following documentation commit only records this link and checklist status.
- Kaggle-measured source commit: https://github.com/4spanch1k/soup-kernel-challenge/commit/b587c5cc2ed5cc2d9457ffc98677cd51cd0beb16. The submission's `solution/` is byte-identical to this commit and `29184d9`, and matches the solution SHA-256 in the saved Kaggle checks.
- Public Kaggle notebook: https://www.kaggle.com/code/yujio2x/soup-kernel-challenge-t4-final-run?scriptVersionId=363982146 (saved version 1, all checks passed).
- Demo video (up to 3 minutes): TODO — add the public video link.
- Presentation: TODO — add the presentation link.

## Team

Two participants. Telegram handles, roles, GitHub and Kaggle accounts supplied by the captain.

| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| @yujio2x (Telegram) | Captain, backend developer | https://github.com/yujio2x | https://www.kaggle.com/yujio2x | https://t.me/yujio2x |
| @yokosowe (Telegram) | Prompt engineer | https://github.com/4spanch1k | https://www.kaggle.com/apanch1k | — |

## Our kernels

Individual kernel shares are not listed here. The combined share for our kernels was 10.8% of
training time in check 4 of the Kaggle run (steps 3+).

| Kernel (`@triton.jit` name) | File | What it replaces | Share of training time |
|---|---|---|---|
| `head_stats_kernel` | `solution/ce_loss.py` (`solution/head_patch.py` wiring) | Full output logits and loss/metric passes | Not measured individually |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | Qwen2 RMSNorm operations | Not measured individually |
| `rope_kernel` | `solution/rope.py` | Qwen2 rotary embedding operations | Not measured individually |
| `nf4_dequant_kernel` | `solution/nf4_dequant.py` | bitsandbytes nested NF4 dequantization | Not measured individually |
| `add_rmsnorm_fwd_kernel`, `add_rmsnorm_bwd_kernel` | `solution/add_rmsnorm.py`, `solution/layer_tail.py` | Residual adds followed by RMSNorm (post-attention and layer boundaries) | Not measured individually |
| **All our kernels** | — | — | **10.8% combined on Kaggle** |

The starter kit's `solution/swiglu.py` example remains active but is not our kernel and is not
counted. The online softmax idea is from Milakov and Gimelshein (2018); the chunked linear
cross-entropy approach was informed by Liger-Kernel and Apple's Cut Cross-Entropy; the NF4/QLoRA
format follows Dettmers et al. (2023) and bitsandbytes. We wrote the code here and do not import
or enable those kernel libraries. Soup's `stream_loss_chunk` was read to preserve its outputs but
is not enabled; only its `STEP_ATTR` hand-off name is reused for the rewind recorder.

## Results

### Kaggle T4, saved version 1

One base and one solution run on 30 October 2026, commit `b587c5c`, using the unchanged harness.

| Metric | Base Soup | Ours | Ratio / change |
|---|---:|---:|---:|
| Whole `soup train`, seconds | 87.77 | 62.10 | 1.413x speedup |
| Peak GPU memory, MiB | 14,345 | 5,441 | 2.636x lower |
| Eval loss | 2.04964 | 2.04952 | -0.006% vs base |

- [x] All 10 harness checks passed on Kaggle T4.
- [x] 49 solution tests and 79 Soup regression tests passed.
- [x] Public saved Kaggle notebook, pinned to `b587c5c` per the finish checklist.
- [x] Own-kernel share 10.8%; first-step loss gap 0.03%; mean loss gap 0.04%.

Evidence: the public notebook above, `docs/kaggle/result.json`, `docs/kaggle/checks.json`
and `docs/kaggle/version-1-log.txt`. Both the result hash and solution hash were verified
against the values in `checks.json` after download.

### Earlier Colab T4 measurements

One base run and one solution run, measured by `harness/run_task.py` and checked by
`harness/check_rules.py` on a Google Colab Tesla T4, branch `m/embed16`, commit `4749158`.
These are Colab results, not Kaggle results.

| Metric | Base Soup | Ours | Ratio / change |
|---|---:|---:|---:|
| Whole `soup train`, seconds | 76.4 | 63.5 | 1.203x speedup in this pair |
| Peak GPU memory, MiB | 14,345 | 5,441 | 2.636x lower |
| Eval loss | 2.04965 | 2.04953 | -0.006% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed on the Colab T4 run.
- [x] `README.md` rewritten for our solution: what we did, how to run it, results and limits.

Check 4: 9.6% of training time. Check 6's loss-curve gap was 0.03% on the first step and 0.04% on
average. Check 8 measured time and memory at 0.831x and 0.379x of base. Soup's 79 regression
tests passed with `apply()` active; all 49 solution tests passed on the T4. Across development
runs the speedup against the same-session base was 1.07x-1.24x.

Repeat check, another seed and two runs per side (jury-like): all 10 checks pass; 80.8 s -> 66.2 s (1.221x), 14,339 -> 5,419 MiB (2.646x), eval loss +0.016%, kernels 8.8%.

## Presentation covers

Describe the optimized operations and kernel ideas, base-versus-solution time and memory, eval
loss, the limits of one Kaggle pair on public data, and next steps. Video and presentation
remain deferred by the team and must be linked before final submission.

## Limitations

Kaggle has one measured run per side. The jury's hidden rows and two runs per side may change
the results. Check 4 passed by 5.8 percentage points in the Kaggle run. MLP recomputation remains
off: an earlier Colab experiment cost 16% time for 27% less memory. RMSNorm outputs fp16 under
autocast, with consumer gradients summed in fp16; Soup's GradScaler is enabled and the measured
mean loss-curve gap was 0.04%. The kernels use Triton 3.x features and PyTorch 2.4 or newer APIs.
