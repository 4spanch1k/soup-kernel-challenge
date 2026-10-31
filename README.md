# Soup Kernel Challenge: mononyx

**Experimental branch `d/activation-codec`:** new T4 runs reached 4339 MiB with
all 10 checks passing and 3811 MiB in an int4 screen. These storage modes are
still off by default; a fixed PEFT dispatch path is awaiting measurement. See
[the experiment report](docs/performance/README.md) for exact flags and evidence.

We optimized the fixed `soup train` run with our own Triton kernels. The run remains Soup CLI
0.77.0 with Qwen2.5-1.5B-Instruct, NF4 weights, LoRA r=16 on `q_proj` and `v_proj`, and fp16 on a
Tesla T4. The fixed task uses batch 4, up to 512 tokens and 64 steps. We did not change the model,
data, batches, training steps, optimizer, schedule, LoRA settings, or Soup switches.

**Team repository:** [BAITC-Hacks/hack-b6e5a4cf-mononyx](https://github.com/BAITC-Hacks/hack-b6e5a4cf-mononyx).

**Submission code snapshot:** [a412248](https://github.com/BAITC-Hacks/hack-b6e5a4cf-mononyx/commit/a412248dcf19b1bbfd13ebf7f93112eb0322abe2).

**Public Kaggle notebook:** [saved version 1, with results](https://www.kaggle.com/code/yujio2x/soup-kernel-challenge-t4-final-run?scriptVersionId=363982146).
The notebook pins [b587c5c](https://github.com/4spanch1k/soup-kernel-challenge/commit/b587c5cc2ed5cc2d9457ffc98677cd51cd0beb16), as specified in `docs/FINISH_CHECKLIST.md`.

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

### Kaggle Tesla T4, 30 October 2026

Saved version 1 ran the imported notebook with **Save & Run All**, Internet enabled and the
GPU T4 x2 accelerator selected. The unchanged harness restricts each measured process to one
GPU. One base and one solution run completed in the same session, using commit `b587c5c`.

| Metric | Base Soup | Ours | Result |
|---|---:|---:|---:|
| Whole `soup train`, seconds | 87.77 | 62.10 | 1.413x speedup |
| Peak GPU memory, MiB | 14,345 | 5,441 | 2.636x lower |
| Eval loss | 2.04964 | 2.04952 | -0.006% vs base |

**All 10 harness checks passed.** All 49 solution tests and all 79 Soup regression tests
passed. Own kernels accounted for 10.8% of training time (steps 3+). The loss-curve gap was
0.03% on the first step and 0.04% on average. Check 5 confirmed the task and Soup outputs within
tolerance; check 9 confirmed a clean measurement. The complete saved notebook took 6m 3s,
including setup, tests and evaluation; that duration is separate from the measured training time.

Evidence: [saved Kaggle logs](https://www.kaggle.com/code/yujio2x/soup-kernel-challenge-t4-final-run/log?scriptVersionId=363982146),
[`result.json`](docs/kaggle/result.json), [`checks.json`](docs/kaggle/checks.json)
and [`version-1-log.txt`](docs/kaggle/version-1-log.txt).
The downloaded result matches the SHA-256 recorded in `checks.json`; the local `solution/`
also matches its recorded solution SHA-256.
These are a single measured pair, not the jury's hidden-data result.

### Earlier Colab Tesla T4 measurements

These are one base run and one solution run from `harness/run_task.py`, followed by
`harness/check_rules.py`, on a Google Colab Tesla T4. The measured solution was branch
`m/embed16`, commit `4749158` (`solution/` of this repository is that code).

| Metric | Base Soup | Ours | Result |
|---|---:|---:|---:|
| Whole `soup train`, seconds | 76.4 | 63.5 | 1.203x speedup in this pair |
| Peak GPU memory, MiB | 14,345 | 5,441 | 2.636x lower |
| Eval loss | 2.04965 | 2.04953 | -0.006% vs base |

All 10 harness checks passed on that Colab T4 run. Our kernels accounted for 9.6% of training time
(check 4). The loss-curve gap was 0.03% on the first step and 0.04% on average (check 6). Time
and memory were 0.831x and 0.379x of base (check 8). Soup's 79 regression tests passed with
`apply()` active (check 10), and all 49 solution tests passed on the T4.

How we got there on the same machine (one run each; Colab speed drifts between sessions, base time
was 75.9 s to 80.9 s, so compare rows by their speedup against the base of the same run):

| Step | Time, s (base) | Peak, MiB | Check 4 |
|---|---:|---:|---|
| Chunked CE + RMSNorm + RoPE | 75.6 (80.6) | 8,427 | 2.9% (fail) |
| + own NF4 dequantization in backward | 75.7 (80.7) | 8,427 | 4.97% (fail) |
| + RMSNorm writes fp16, fused add + post-attention norm | 65.8 (77.5) | 6,321 | 6.4% |
| + fused layer boundaries, own NF4 dequantization in the matmul forward | 68.2 (80.9) | 6,321 | 8.8% |
| + packed layers: padding rows skipped (43% of rows are padding) | 61.1 (75.9) | 6,321 | 10.4% |
| + tied embedding and head compacted into one exact fp16 table (final) | 63.5 (76.4) | 5,441 | 9.6% |

Two patches that are not kernels: `solution/layer_tail.py` runs the decoder stack on the valid
tokens only and keeps the padded layout just for attention (the rules allow removing padding inside
a batch), and `solution/embed16.py` replaces the tied fp32 embedding/head (exactly fp16-representable
because the model is loaded in fp16) by one shared fp16 parameter; the embedding output stays fp32.
The memory peak is set by the densest batch, which has almost no padding, so packing helps time but
not peak memory.

MLP recomputation (`solution/recompute.py`, our own `torch.utils.checkpoint`, not Soup's switch) was
measured on the T4 in one session, solution only: off 59.7 s / 6,321 MiB, on 69.3 s / 4,601 MiB. It
costs 16% time for 27% memory, so it stays off.

### Repeat with another seed, two runs per side (the way the jury runs it)

`python harness/run_task.py --repeats 2 --seed 2026`, then `check_rules.py`, on the same code (commit
`29184d9`), Google Colab Tesla T4. **All 10 checks passed**: base 80.8 s / 14,339 MiB, ours 66.2 s /
5,419 MiB (1.221x faster, 2.646x less memory), eval loss 2.05134 -> 2.05167 (+0.016%, limit +0.5%),
our kernels 8.8% of training time in the lowest run, loss-curve gap 0.00-0.01% (first step) and
0.01-0.05% (mean) in all four extra runs. This also passed on a second seed; it does not establish
performance on the jury's hidden rows.

## Limits

- Kaggle results are one measured run per side on public data. Earlier Colab measurements,
  including a two-repeat check with another seed, are reported separately above.
- The jury uses hidden rows and two runs per side, so results can move. Check 4 had 5.8
  percentage points of margin in this Kaggle run (10.8% against the 5% requirement).
- `solution/recompute.py` can recompute MLP activations; it is off by default because it cost 16% time for 27% memory in our measurement.
- RMSNorm writes fp16 under autocast. Gradients from its several consumers are summed in fp16;
  Soup's GradScaler is enabled, and the observed mean loss-curve gap was 0.04%.
- The kernels use Triton 3.x features including `tl.interleave` and `tl.argmax` in loops, and
  PyTorch APIs available in torch 2.4 or newer.

## How to run

The submission repository is
[BAITC-Hacks/hack-b6e5a4cf-mononyx](https://github.com/BAITC-Hacks/hack-b6e5a4cf-mononyx).
The public development repository below supplies the exact measured commit used by the saved
Kaggle notebook. `solution/` is unchanged in the submission copy.
For Kaggle, open the saved notebook above, or import `notebook/soup_kernel_challenge.ipynb`,
select T4 with Internet enabled, set `COMMIT = "b587c5c"` and use Save & Run All.
Or clone the public development repository in one notebook cell:

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
