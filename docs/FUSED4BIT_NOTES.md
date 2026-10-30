# Fused NF4 GEMM bake-off

## What the experiment does

`solution/fused4bit.py` decodes the existing nested NF4 representation while a Triton matrix tile is
being loaded, then runs fp16 `tl.dot` with fp32 accumulation. The forward computes
`X[M,K] @ W[N,K].T`; backward computes `dY[M,N] @ W[N,K]`. Both return fp16. Bias is added in the
forward epilogue. Scale values follow the existing decoder exactly: the high nibble is the first
weight, each 64 values share one uint8 scale code, and that code is expanded by the corresponding
256-scale absmax and offset.

The fixed tiles are `BLOCK_K=64`, `BLOCK_N=64`, and `BLOCK_M=32` for `M < 512` or 64 otherwise,
with four warps and one stage. K blocks therefore begin and end at NF4 scale boundaries. The M
split supplies more independent blocks for the 256-wide K/V output than a large M tile would; the
wide MLP outputs already provide ample N parallelism. No autotuning or generated external kernel
is used.

The patch wraps bitsandbytes `MatMul4Bit.forward` and `.backward` only when `ENABLED = True` or
`SOUP_FUSED4BIT=1`. It selects the fused path only for CUDA fp16 activations, 2-D nested NF4
weights, NF4 blocksize 64, nested scale blocksize 256, and K divisible by 64. Other inputs use the
existing method. Default runs remain on the reference path.

## Expected performance, before the T4 run

At M=800, one idealized GEMM loads each activation once and writes each output once. Counting the reference's NF4
byte read, fp16 dequantized-weight write, and fp16 GEMM weight read gives 4.5 bytes per weight
element. Fused reads the packed weight and one scale code per 64 values, about 0.516 bytes per
element before cache-line effects. This cuts the *weight path* traffic by about 88%; it does not
remove the activation reads, output writes, scale decode, or any matrix-multiply FLOPs.

| Weight shape `[N,K]` | FLOPs at M=800 | Ideal reference traffic | Ideal fused traffic | Arithmetic intensity, reference → fused |
|---|---:|---:|---:|---:|
| 1536×1536 (q/o) | 3.77 GFLOP | 15.5 MB | 6.1 MB | 243 → 616 FLOP/byte |
| 256×1536 (k/v) | 0.63 GFLOP | 4.64 MB | 3.07 MB | 136 → 205 FLOP/byte |
| 8960×1536 (gate/up) | 22.0 GFLOP | 78.7 MB | 23.9 MB | 280 → 922 FLOP/byte |
| 1536×8960 (down) | 22.0 GFLOP | 78.7 MB | 23.9 MB | 280 → 922 FLOP/byte |

These are lower-bound traffic estimates: they assume a weight tile is fetched only once. The Triton
kernel reloads the packed tile for each M tile, just as GEMM implementations reload tiles at some
level of the hierarchy. Cache behavior and decode instruction cost decide how much of the ideal
88% survives. The comparison starts from the measured 350 ms of cuBLAS GEMMs and 31 ms of NF4
dequantization in a 565 ms step. If the fused GEMMs ran at exactly cuBLAS speed and removed all
31 ms of standalone dequantization, the step ceiling from this path alone would be
`565 / 534 = 1.058x`. Fused GEMMs can be up to about 8.9% slower than cuBLAS before their combined
GEMM-plus-dequant time loses to the measured 381 ms reference path. Compilation is also charged in
the full run.

T4 is sm75 and supports fp16 Tensor Core MMA (`mma.sync.m16n8k8`); `tl.dot` can target that
instruction family. cuBLAS already uses larger, hand-tuned Turing tiles (reported as 256×128 or
128×128 here). Our smaller 32/64-row tiles, scale loads, nibble unpacking, and layout conversion
may use Tensor Cores yet still lose to cuBLAS. The benchmark therefore decides whether this path is
kept. No speedup is claimed before that run.

## Alternatives considered

- **Dequantize once and keep fp16 weights through backward:** this removes repeated decoding but
  retains roughly 2.62 GB of fp16 weights for the 28 layers, in addition to the packed weights.
  Adding that copy to the measured 14,345 MiB peak would exceed the T4's 15 GiB. Keeping only
  selected layers would complicate lifetime management and still needs a separate measured memory
  budget.
- **Repack weights once into a Tensor Core layout:** this could improve coalescing, especially for
  the backward orientation, but a second packed copy costs about 0.66 GB and a fp16 copy costs
  about 2.62 GB. Forward and backward prefer opposite physical orientations, so one layout may not
  serve both. It also adds load-time work and another representation to validate.
- **Convert to int8:** it doubles the packed weight payload and requires a scaling/rounding scheme.
  That risks changing the NF4-dequantized values which the task requires us to preserve, and adds
  conversion work that this bake-off is trying to remove.
- **Swizzle nested scales:** arranging scale codes for the 64×64 tiles could reduce scale address
  overhead, but would require an extra transformed scale representation. First measure whether
  the direct scale loads are the limiting cost.

## Measurement

On the T4, run `python scripts/bench_gemm4bit.py`. It quantizes all four real matrix shapes with
bitsandbytes, measures forward and input-gradient forms for M=400, 826, 1400, and 2048 with CUDA
events, and prints first-call wall time separately. That first-call number includes the Triton
compile when the specialization misses the cache, plus its launch. The table reports median
reference/fused latency, speedup, maximum absolute error, and maximum relative error. Each shape
class gets a median verdict across all M values and directions.

The tests compare all four matrix shapes at M=1, 17, and 826 and exercise gradients through
`MatMul4Bit`. Run `python -m pytest solution/test_fused4bit.py -q`. The training A/B command is
`SOUP_FUSED4BIT=1 bash scripts/kaggle_3_full.sh`; leave the flag unset for the reference run. There
is no CUDA device in the local development environment, so kernel correctness and performance
remain **not run** until the owner runs them on T4.
