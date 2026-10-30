# Submission: TODO — team name

## Links

- Repository: https://github.com/4spanch1k/soup-kernel-challenge
- Final commit: TODO — add the final commit SHA and commit URL.
- Public Kaggle notebook: TODO — add the saved public Kaggle notebook URL (not yet available).
- Demo video (up to 3 minutes): TODO — add the public video link.
- Presentation: TODO — add the presentation link.

## Team

TODO — add team members, roles, GitHub/Kaggle accounts, and captain contact.

| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO |

## Our kernels

Individual kernel shares were not recorded. The combined share for our kernels was 6.4% of
training time in check 4.

| Kernel (`@triton.jit` name) | File | What it replaces | Share of training time |
|---|---|---|---|
| `head_stats_kernel` | `solution/ce_loss.py` (`solution/head_patch.py` wiring) | Full output logits and loss/metric passes | Not measured individually |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | Qwen2 RMSNorm operations | Not measured individually |
| `rope_kernel` | `solution/rope.py` | Qwen2 rotary embedding operations | Not measured individually |
| `nf4_dequant_kernel` | `solution/nf4_dequant.py` | bitsandbytes nested NF4 dequantization | Not measured individually |
| `add_rmsnorm_fwd_kernel`, `add_rmsnorm_bwd_kernel` | `solution/add_rmsnorm.py` | Post-attention residual add and RMSNorm | Not measured individually |
| **All our kernels** | — | — | **6.4% combined** |

The starter kit's `solution/swiglu.py` example remains active but is not our kernel and is not
counted. The online softmax idea is from Milakov and Gimelshein (2018); the chunked linear
cross-entropy approach was informed by Liger-Kernel and Apple's Cut Cross-Entropy; the NF4/QLoRA
format follows Dettmers et al. (2023) and bitsandbytes. We wrote the code here and do not import
or enable those kernel libraries. Soup's `stream_loss_chunk` was read to preserve its outputs but
is not enabled; only its `STEP_ATTR` hand-off name is reused for the rewind recorder.

## Results

One base run and one solution run, measured by `harness/run_task.py` and checked by
`harness/check_rules.py` on a Google Colab Tesla T4, branch `f/add-rmsnorm`, commit `6796e4c`.
These are Colab results, not Kaggle results.

| Metric | Base Soup | Ours | Ratio / change |
|---|---:|---:|---:|
| Whole `soup train`, seconds | 77.5 | 65.8 | 1.176x speedup in this pair |
| Peak GPU memory, MiB | 14,345 | 6,321 | 2.269x lower |
| Eval loss | 2.04966 | 2.04948 | -0.009% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed on the Colab T4 run.
- [x] `README.md` rewritten for our solution: what we did, how to run it, results and limits.

Check 6's loss-curve gap was 0.01% on the first step and 0.04% on average. Check 8 measured time
and memory at 0.850x and 0.441x of base. Soup's 79 regression tests passed with `apply()` active;
all 37 solution tests passed on the T4. Across development runs and base-run variation, the
observed speed gain was roughly 1.07x–1.18x.

## Presentation covers

Describe the optimized operations and kernel ideas, base-versus-solution time and memory, eval
loss, the Colab-versus-Kaggle measurement limit, and next steps.

## Limitations

The measurements are one run per side on Colab, not Kaggle. The team's Kaggle account lacks phone
verification for GPU and Internet access; the jury's hidden rows and two runs per side may change
the results. Check 4 passed by about 1.4 percentage points. MLP recomputation is off by default
and was not measured. RMSNorm outputs fp16 under autocast, with consumer gradients summed in
fp16; Soup's GradScaler is enabled and the mean loss-curve gap was 0.04%. The kernels use Triton
3.x features and PyTorch 2.4 or newer APIs.
