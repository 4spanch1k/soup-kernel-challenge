# Soup Kernel Challenge: Our Solution

We optimized the fixed `soup train` run with our own Triton kernels. The run remains Soup CLI
0.77.0 with Qwen2.5-1.5B-Instruct, NF4 weights, LoRA r=16 on `q_proj` and `v_proj`, and fp16 on a
Tesla T4. The fixed task uses batch 4, up to 512 tokens and 64 steps. We did not change the model,
data, batches, training steps, optimizer, schedule, LoRA settings, or Soup switches.

**Public Kaggle notebook:** TODO — add the saved public Kaggle notebook URL (not yet available).

## What we changed

A shape-recording profile showed that the residual stream stays in fp32: `prepare_model_for_kbit_training`
keeps the embedding in fp32, and bitsandbytes `Linear4bit` returns its result in the input dtype.
That causes fp16 linear outputs to be converted back to fp32 and keeps activations in fp32; casts
and adds took about 17% of GPU time in that profile. We therefore kept the residual stream's
required fp32 values while writing normalized activations directly in the autocast dtype.

The output head also creates a very large vocabulary-logit tensor. Our head processes token rows
in chunks, calculates Soup's loss and logged statistics without materializing all logits, and
keeps the gradients for labelled tokens only. RMSNorm, RoPE, NF4 dequantization, and the
residual adds fused with RMSNorm are also handled by Triton. RMSNorm, RoPE and the fused
add-plus-norm each have forward and backward kernels; NF4 dequantization replaces bitsandbytes'
dequantization both in the 4-bit matmul forward and in its backward.

| Kernel | File | Replaces | Key idea |
|---|---|---|---|
| `head_stats_kernel` | `solution/ce_loss.py`, wired by `solution/head_patch.py` | Full output logits and their loss/metric passes | Project 256 token rows at a time; online log-sum-exp computes cross-entropy, entropy and argmax, then reuses the chunk for the loss gradient. Padding rows are dropped; the hidden gradient is computed chunk by chunk, with the loss scale applied in backward. |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | Qwen2 RMSNorm's sequence of PyTorch operations | Normalize one row per program with fp32 statistics and save only one reciprocal standard deviation per row. Under autocast, write the consumer dtype directly. |
| `rope_kernel` | `solution/rope.py` | Qwen2's separate slice, rotate, multiply and add operations for Q and K | Rotate both halves of a head in one forward kernel; apply the inverse rotation in backward. |
| `nf4_dequant_kernel` | `solution/nf4_dequant.py` | bitsandbytes' dequantization (`dequantize_4bit` in backward; the dequantize step of `gemm_4bit` in `MatMul4Bit.forward`) for supported nested NF4 weights | Decode packed nibbles and nested scales with the NF4 lookup table, interleave the two nibbles and write the dequantized fp16 weights in one contiguous store. |
| `add_rmsnorm_fwd_kernel`, `add_rmsnorm_bwd_kernel` | `solution/add_rmsnorm.py`, `solution/layer_tail.py` | Each residual add followed by an RMSNorm: the post-attention add + norm, and the MLP add + the next layer's input norm (or the final norm) | Add and normalize in one forward kernel, retaining the fp32 residual and writing the normalized value in the consumers' autocast dtype; combine both gradient paths in backward. `layer_tail.py` (a replacement of `Qwen2Model.forward` for the training case, with fallback to the original) applies it at layer boundaries. |

The starter kit's `solution/swiglu.py` example remains active. It is not our work and is not
included in our kernel share.

### Sources and implementation boundary

- The online softmax calculation follows the idea in Milakov and Gimelshein's 2018 paper,
  [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867).
- The chunked linear cross-entropy approach was informed by
  [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) and Apple's
  [Cut Cross-Entropy](https://github.com/apple/ml-cross-entropy). Our kernels are written in this
  repository; neither project is imported or enabled.
- The NF4/QLoRA format follows Dettmers et al.,
  [QLoRA (2023)](https://arxiv.org/abs/2305.14314), and the
  [bitsandbytes source](https://github.com/bitsandbytes-foundation/bitsandbytes).
- We read Soup's `stream_loss_chunk` implementation to identify the outputs that must be
  preserved. We do not enable that Soup switch. The rewind recorder's `STEP_ATTR` name is reused
  only to hand it our per-token losses.

## Results

These are one base run and one solution run from `harness/run_task.py`, followed by
`harness/check_rules.py`, on a Google Colab Tesla T4. The measured solution was branch
`h/nf4-forward`, commit `1626fe6` (the final code of this repository).

| Metric | Base Soup | Ours | Result |
|---|---:|---:|---:|
| Whole `soup train`, seconds | 80.9 | 68.2 | 1.185x speedup in this pair |
| Peak GPU memory, MiB | 14,345 | 6,321 | 2.269x lower |
| Eval loss | 2.04966 | 2.04949 | -0.008% vs base |

All 10 harness checks passed on that Colab T4 run. Our kernels accounted for 8.8% of training time
(check 4). The loss-curve gap was 0.03% on the first step and 0.04% on average (check 6). Time
and memory were 0.844x and 0.441x of base (check 8). Soup's 79 regression tests passed with
`apply()` active (check 10), and all 42 solution tests passed on the T4.

How we got there on the same machine (one run each; base time varied between 77.5 s and 80.9 s):

| Step | Time, s | Peak, MiB | Check 4 |
|---|---:|---:|---|
| Chunked CE + RMSNorm + RoPE | 75.6 | 8,427 | 2.9% (fail) |
| + own NF4 dequantization in backward | 75.7 | 8,427 | 4.97% (fail) |
| + RMSNorm writes fp16, fused add + post-attention norm | 65.8 | 6,321 | 6.4% |
| + fused layer boundaries (`layer_tail.py`) | 68.4 | 6,321 | 6.0% |
| + own NF4 dequantization in the matmul forward (final) | 68.2 | 6,321 | 8.8% |

The layer-boundary fusion did not change the time measurably against run-to-run noise; it stays
because the final configuration passed all checks with the largest margin. Observed speed gains
over several base runs are roughly 1.07x to 1.19x, not a stable precise ratio.

## Limits

- These results are from Colab, not Kaggle. The team's Kaggle account lacks phone verification
  required for GPU and Internet access, so a Kaggle run has not been completed and may differ.
- Each side has one measured run. The jury uses hidden rows and two runs per side; results can
  move. Check 4 passed by about 3.8 percentage points in the final run.
- `solution/recompute.py` can recompute MLP activations but is off by default and was not measured.
- RMSNorm writes fp16 under autocast. Gradients from its several consumers are summed in fp16;
  Soup's GradScaler is enabled, and the observed mean loss-curve gap was 0.04%.
- The kernels use Triton 3.x features including `tl.interleave` and `tl.argmax` in loops, and
  PyTorch APIs available in torch 2.4 or newer.

## How to run

The repository is public at
[github.com/4spanch1k/soup-kernel-challenge](https://github.com/4spanch1k/soup-kernel-challenge).
For Kaggle, import `notebook/soup_kernel_challenge.ipynb` and select a T4 GPU with Internet
enabled. Or clone the public repository in one notebook cell:

```python
!git clone https://github.com/4spanch1k/soup-kernel-challenge.git /kaggle/working/repo
%cd /kaggle/working/repo
!bash scripts/kaggle_1_tests.sh
```

Then run `!bash scripts/kaggle_2_quick.sh` for a solution-only profiled run, or
`!bash scripts/kaggle_3_full.sh` for base-versus-solution measurement, all 10 checks and the
score. The test script installs the pinned packages, prepares the public data and runs both test
suites.

For Colab, select a T4 GPU and run this setup in a code cell:

```python
!git clone https://github.com/4spanch1k/soup-kernel-challenge.git
%cd soup-kernel-challenge
!python -m pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
!python task/prepare_data.py
```

For local Linux with CUDA, run the equivalent shell setup from the repository root after cloning:

```bash
python -m pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
```

Run the kernel tests and Soup regression tests:

```bash
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
```

Run a quick solution-only measurement, or the full base-versus-solution measurement and checks:

```bash
python harness/run_task.py --only solution --out runs_try
python harness/run_task.py && python harness/check_rules.py
```

The harness writes `runs/result.json` and `runs/checks.json`. Results count for the challenge only
on the jury's T4 environment.

## License

Apache-2.0, see `LICENSE` and `NOTICE`. The five test files come from Soup v0.77.0 (Apache-2.0).
Data: [AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2)
(MIT on its card), a Kazakh translation of [Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
(CC BY-NC 4.0); it is downloaded at run time for this benchmark. Model:
[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0).
