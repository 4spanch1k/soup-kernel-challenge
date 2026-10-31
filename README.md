# Soup Kernel Challenge: mononyx

We optimized the fixed `soup train` run with our own Triton kernels. The run remains Soup CLI
0.77.0 with Qwen2.5-1.5B-Instruct, NF4 weights, LoRA r=16 on `q_proj` and `v_proj`, and fp16 on a
Tesla T4. The fixed task uses batch 4, up to 512 tokens and 64 steps. We did not change the model,
data, batches, training steps, optimizer, schedule, LoRA settings, or Soup switches.

**Result in one line (Colab Tesla T4, base and solution in the same session, final code `d881159`):** 77.3 s → 54.1 s (1.428x faster), 14,345 → 4,591 MiB (3.125x less memory), eval loss 2.04966 → 2.04987 (+0.010%), all 10 harness checks passed, own kernels 13.1% of training time.
The earlier saved Kaggle run (87.77 s → 62.10 s, 14,345 → 5,441 MiB, 10/10 checks) measured the code before the final fix described below.

**Team repository:** [BAITC-Hacks/hack-b6e5a4cf-mononyx](https://github.com/BAITC-Hacks/hack-b6e5a4cf-mononyx).

**Submission code snapshot (taken before the final fix; the final code is `d881159`, see below):** [a412248](https://github.com/BAITC-Hacks/hack-b6e5a4cf-mononyx/commit/a412248dcf19b1bbfd13ebf7f93112eb0322abe2).

**Public Kaggle notebook:** [saved version 1, with results](https://www.kaggle.com/code/yujio2x/soup-kernel-challenge-t4-final-run?scriptVersionId=363982146).
The notebook shows the earlier code, before the final fix, and pins [b587c5c](https://github.com/4spanch1k/soup-kernel-challenge/commit/b587c5cc2ed5cc2d9457ffc98677cd51cd0beb16), as specified in `docs/FINISH_CHECKLIST.md`.

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

### Final update: packed-row path fix (Colab Tesla T4, 31 October 2026)

After the Kaggle run we found that the patched `Qwen2Model.forward` always took its original
fallback: PEFT passes `output_attentions=None`, `output_hidden_states=None` and `return_dict=None`
down to the base model, and the wrapper treated those default-valued kwargs as unsupported. The
fix is limited to `solution/layer_tail.py`: it drops only those default-valued flags before the
check for unsupported kwargs, so the packed-row path (padding rows skipped through the decoder
layers, padded layout kept only for attention) now runs. Final source commit: `d881159dadf73b09fb847420a70ac0644cedc429`; `solution/`
differs from the earlier measured code in that file only.

One base and one solution run in the same session with the unchanged harness:

| Metric | Base Soup | Ours | Result |
|---|---:|---:|---:|
| Whole `soup train`, seconds | 77.3 | 54.1 | 1.428x speedup |
| Peak GPU memory, MiB | 14,345 | 4,591 | 3.125x lower |
| Eval loss | 2.04966 | 2.04987 | +0.010% vs base |

All 10 harness checks passed; own kernels accounted for 13.1% of training time. A second, independent paired run of the same commit on a Kaggle Tesla T4 (interactive notebook session, not a saved
version) gave base 98.1 s, ours 58.5 s (1.677x), 14,345 → 4,591 MiB (3.125x less), eval loss 2.04965 → 2.04983 (+0.009%),
all 10 checks passed. Each run is a single pair; the saved public Kaggle notebook was not re-run on this commit.

### Kaggle Tesla T4, 30 October 2026 (code before the final fix)

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

### Earlier Colab Tesla T4 measurements (pre-fix code, historical)

These are one base run and one solution run from `harness/run_task.py`, followed by
`harness/check_rules.py`, on a Google Colab Tesla T4; a two-run repeat with another seed is
reported further below. The measured solution was branch `m/embed16`, commit `4749158`; the
`solution/` directory is byte-identical in that commit, in `29184d9` and in the Kaggle-measured
`b587c5c`.

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
was 75.9 s to 80.9 s, so compare rows by their speedup against the base of the same run; the
speedup against the same-session base grew from about 1.07x in the first row to 1.24x in the
packed-layers row):

| Step | Time, s (base) | Peak, MiB | Check 4 |
|---|---:|---:|---|
| Chunked CE + RMSNorm + RoPE | 75.6 (80.6) | 8,427 | 2.9% (fail) |
| + own NF4 dequantization in backward | 75.7 (80.7) | 8,427 | 4.97% (fail) |
| + RMSNorm writes fp16, fused add + post-attention norm | 65.8 (77.5) | 6,321 | 6.4% |
| + fused layer boundaries, own NF4 dequantization in the matmul forward | 68.2 (80.9) | 6,321 | 8.8% |
| + packed-layer path (pre-fix: PEFT's default-valued flags sent the call to the original path, so padding rows were not actually skipped) | 61.1 (75.9) | 6,321 | 10.4% |
| + tied embedding and head compacted into one exact fp16 table (final) | 63.5 (76.4) | 5,441 | 9.6% |

Two patches that are not kernels: `solution/layer_tail.py` runs the decoder stack on the valid
tokens only and keeps the padded layout just for attention (the rules allow removing padding inside
a batch), and `solution/embed16.py` replaces the tied fp32 embedding/head (exactly fp16-representable
because the model is loaded in fp16) by one shared fp16 parameter; the embedding output stays fp32.
In these pre-fix runs the peak stayed at 5,441 MiB while `layer_tail.py` fell back to the original path. After the
fix in `d881159` activated packed-row execution, the measured peak was 4,591 MiB in both reported final pairs
(each a single paired run).

MLP recomputation (`solution/recompute.py`, our own `torch.utils.checkpoint`, not Soup's switch) was
measured on the T4 in one session, solution only: off 59.7 s / 6,321 MiB, on 69.3 s / 4,601 MiB. It
costs 16% time for 27% memory, so it stays off.

### Earlier Colab repeat with another seed, two runs per side (the way the jury runs it)

`python harness/run_task.py --repeats 2 --seed 2026`, then `check_rules.py`, on the same code (commit
`29184d9`), Google Colab Tesla T4. **All 10 checks passed**: base 80.8 s / 14,339 MiB, ours 66.2 s /
5,419 MiB (1.221x faster, 2.646x less memory), eval loss 2.05134 -> 2.05167 (+0.016%, limit +0.5%),
our kernels 8.8% of training time in the lowest run, loss-curve gap 0.00-0.01% (first step) and
0.01-0.05% (mean) in all four extra runs. This also passed on a second seed; it does not establish
performance on the jury's hidden rows.

## Limits

- Kaggle gives one measured run per side on public data. The earlier Colab measurements are
  reported separately above: one run per side, plus a two-run repeat with another seed. None of
  them is the jury's hidden-data result.
- The jury uses hidden rows and two runs per side, so results can move. Check 4 (own kernels at
  least 5% of training time) measured 10.8% on Kaggle, 9.6% in the earlier Colab run and 8.8% in
  the lowest run of the Colab seed repeat; on Kaggle that is 5.8 percentage points above the
  requirement.
- `solution/recompute.py` can recompute MLP activations; it is off by default because it cost 16%
  time for 27% memory in our measurement.
- RMSNorm writes fp16 under autocast. Gradients from its several consumers are summed in fp16;
  Soup's GradScaler is enabled, and the observed mean loss-curve gap was 0.04%.
- Tested environment (Kaggle, from `docs/kaggle/README.md`): Python 3.13.15, torch 2.11.0+cu128,
  Triton 3.6.0, Tesla T4. The kernels use Triton 3.x features including `tl.interleave` and
  `tl.argmax` in loops; we have not run them on other torch or Triton versions.

## Tried and rejected

Measured on a Colab T4 with the full harness, each against its own base from the same session
(details in `docs/CODEX_VS_CLAUDE.md` and `docs/HANDOFF.md`). None of these is in `solution/`.

| Idea | Measured result | Decision |
|---|---|---|
| MLP recomputation (`solution/recompute.py`) | off 59.7 s / 6,321 MiB, on 69.3 s / 4,601 MiB: 16% time for 27% memory | stays off |
| Per-sample causal attention on packed rows (`q/varlen-attn`) | 66.9 / 68.3 s against 61.8 / 68.0 s for `main` (two alternating runs each); peak memory the same, 5,441 MiB | no gain, not merged |
| NF4 GEMM as one Triton kernel (`c2/fused4bit`) | 10-200 times slower than dequantization plus cuBLAS | rejected |
| Fused LoRA path (`l/lora-fuse`) | patch never fired (PEFT uses its own 4-bit LoRA class); 63.2 s / 5,441 MiB, same as `main`; micro-benchmark 1.26 ms against 0.94 ms per q call | rejected |
| Shorter start-up (`s/startup`) | cold Triton compile of all our kernels is 4.7 s; a warm-up thread could hide it but risks reading as background work | not used |
| Faster model loading (`w/load-phase`) | no value-preserving saving above 0.5 s | not used |

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
