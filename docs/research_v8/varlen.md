# PyTorch varlen attention experiment

## What changed

`SOUP_X_VARLEN=1` enables a path inside the packed-token branch of
`solution/layer_tail.py`. For a right-padded batch it gathers the already-packed Q/K/V,
applies RoPE to those rows, expands the two K/V heads to the twelve query heads, then calls
PyTorch's `_efficient_attention_forward` once per transformer layer with cumulative sequence
lengths and causal top-left masking. PyTorch's registered derivative sends backward through its
efficient-attention backward operator, as declared in
[PyTorch 2.11's derivatives](https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml#L2539-L2541).
No Triton attention kernel was added.

The default is off. If the mask, device, dtype, sliding-window, dropout, operator schema, autograd
registration, or runtime call is incompatible, that layer uses the existing padded attention path;
after a runtime failure the remaining layers also use that path. A test was added to compare forward
values and Q/K/V gradients with causal SDPA for unequal odd lengths, rows of length one, and equal
row lengths.

CUDA tests and T4 A-B-A-B measurements: **не запускалось** (the development machine has no GPU).

## Estimate and known measurements

- Given profile: attention is 51.7 ms of a 454.6 ms training step (about 11.4%). Across 64 steps,
  3.3 seconds is the absolute ceiling if attention became free; the actual saving must be smaller.
- The previous per-example SDPA experiment measured 61.8 / 68.0 s for A and 66.9 / 68.3 s for B,
  with identical 5,441 MiB peaks. It did not improve runtime and was rejected. A single varlen call
  per layer removes those per-example attention launches, but its gain has not been measured yet.
- The run script prints the measured wall time, peak GPU memory, and eval loss for each A-B-A-B
  solution-only run. Treat those values as measurements only on the GPU session that produced them.

## Acceptance criteria

Keep this option only after a same-session T4 comparison shows all of the following:

1. All 10 `harness/check_rules.py` checks pass on a full base-versus-solution run.
2. Our Triton kernels account for at least 6% of training time, leaving margin above the 5% rule.
3. Peak GPU memory does not exceed the measured `main` run.
4. The wall-time gain is larger than the spread between the paired A runs; a tie or noisy change is
   not a win.
5. Eval loss is no more than 0.1% above base.
6. The varlen attention forward/backward tests pass on a CUDA GPU.

The A-B-A-B script is a solution-only timing experiment; it cannot establish the 10-check or
base-loss criteria by itself. Run the full harness with `SOUP_X_VARLEN=1` after a timing win.

## Run on Kaggle

`notebook/t4_experiments4.ipynb` clones `c8/varlen`, runs the kernel and Soup regression tests, then
runs `scripts/t4_exp4.sh`. The branch must exist on the public repository before the notebook can
check it out. Results are captured in `/kaggle/working/exp4.log`.
