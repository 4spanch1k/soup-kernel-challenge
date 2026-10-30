# Cheap T4 measurements, no code changes (2026-10-30, Colab Tesla T4, main = 6f3a11e)

Run with `bash scripts/t4_measure.sh` (branch `x/measure`). Versions: torch 2.11.0+cu130, triton 3.6.0,
transformers 5.16.1, peft 0.20.0, trl 0.29.1, bitsandbytes 0.50.2, accelerate 1.14.0. All numbers below are MEASURED.

## 1. Triton `tl.dot` on sm75 (scripts/t4_dot_check.py)
Plain Triton fp16 matmul vs `torch.mm` (cuBLAS): 30-560x slower (1.0-1.2 TFLOP/s against 29-46 TFLOP/s); the generated PTX has
`mma` = 0 and 1024-2048 scalar `fma.f32`. So Triton on this T4 does not use tensor cores: our own Triton GEMM or flash-attention
cannot be competitive here. Keep cuBLAS/SDPA for matmuls and attention.

## 2. SDPA backends, shapes of our run: B=4, 12 query heads, 2 KV heads, d=128, T=512, forward+backward per layer (scripts/t4_sdpa_check.py)
| Call | Backend | ms | peak MiB |
|---|---|---:|---:|
| repeat_kv + 4D mask | memory-efficient (cutlass) | 2.94 | 72 |
| repeat_kv + is_causal | memory-efficient | 1.94 | 72 |
| enable_gqa + is_causal | memory-efficient | not available | |
| any | flash | not available on sm75 | |
| any (incl. enable_gqa) | math | 8.5-8.7 | 268 |
Default dispatcher: repeat_kv + mask 2.89 ms; enable_gqa + is_causal 8.38 ms (falls back to math, about 3x slower, about 3.7x the memory).
`is_causal` instead of the 4D mask is about 34% faster for attention.

## 3. Profile of our run (solution only, `--profile`, 24 steps)
* Step: 454.6 ms GPU. cuBLAS GEMMs (`turing_fp16_s1688gemm_*`) about 64%; NF4 dequant 30.6 ms (6.7%);
  attention is the memory-efficient cutlass kernel (`fmha_cutlassF/B_f16_..._sm75`): 14.8 fwd + 36.9 bwd = 51.7 ms (11.4%), NOT the math fallback;
  example SwiGLU 23.2 ms (5.1%, does not count for check 4); fused dropout 2.9 ms, our add+RMSNorm 7.0 ms, RoPE 3.3 ms, head 2.6 ms.
* Solution-only wall 63.9 s, peak 5,441 MiB, eval loss 2.04948; `train_loop_s` 37.14 s, `train_loop_s - sum(step_s)` = 0.47 s (no host idle in the loop).
* torch max allocated 5,107 MiB, max reserved 5,298 MiB, nvidia-smi peak 5,441 MiB: only about 143 MiB above the torch reserved pool,
  so the peak is almost all torch tensors (weights + saved activations), and the allocator gap is about 191 MiB.
* `own_kernel_ms` (ms over the timed loop): nf4_dequant 2,029; rope 373; add_rmsnorm fwd/bwd 211/255; rmsnorm fwd/bwd 108/174; head_stats 173;
  example swiglu fwd/bwd 544/928 (not counted). Own total 3,322 ms of 37,140 ms = 8.9% (check 4 needs >= 5%).

## What this means
* Own Triton attention/GEMM on this T4: dead (no tensor cores in Triton 3.6 on sm75).
* Attention is already the efficient kernel on the real run. Switching the masked call to `is_causal` would cut about 34% of the 51.7 ms,
  about 17 ms/step = 3.7% of the step, about 1.1 s of 63.5 s (an estimate from the microbenchmark, needs a paired A/B run).
* Risk found: a batch without padding makes transformers pass `enable_gqa=True` with no mask, which on this torch falls back to math attention
  (about 3x slower, more memory). Not seen on the public rows; test with an all-512-token batch built only from public rows 0-255.
