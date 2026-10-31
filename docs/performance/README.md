# T4 optimization experiments, 31 October 2026

Experiments on `d/activation-codec` are separate from the published baseline.
Storage modes remain off by default until the selected combination is qualified
through the unchanged harness. Only `solution/` is copied by the jury: notebook
environment flags alone do not enable an optimization in the final submission.

## Completed measurements

Compare time within the same session; GPU speed differs between sessions.

| Run | Candidate | Time, s | Peak, MiB | Eval loss | Validation |
|---|---|---:|---:|---:|---|
| Stage 2 | Warm control | 68.88 | 5441 | 2.04955 | Solution-only |
| Stage 2 | Int8 MLP activations + parallel CE | 69.93 | 4503 | 2.04951 | Solution-only |
| Qualification, seed 2026 | Int8 + requested varlen | 71.18 | 4499 | 2.05174 | 2 repeats per side; 10/10 |
| Qualification, seed 2026 | Int8 + fp16 saved norms + requested varlen | 70.81 | 4339 | 2.05174 | 2 repeats per side; 10/10 |
| Stage 3 | Int8 + fp16 saved norms, warm control | 67.55 | 4323 | 2.04947 | Solution-only |
| Stage 3 | Int8 + fp16 saved norms + startup flags | 64.85 | 4323 | 2.04950 | Solution-only |
| Stage 3 | Int4 + fp16 saved norms + startup flags | 67.61 | 3811 | 2.05089 | 21 codec tests; solution-only |

The two-repeat int8/norm candidate had base 88.39 s / 14339 MiB / 2.05132:
1.2483x speedup, 3.3047x lower memory, eval +0.020%, mean training-loss gap
0.05%, own-kernel share at least 11.7%, and 79 Soup tests passed. This does not
qualify the later int4 or dispatch changes, which need their own complete run.
The original downloaded result matches the checker's SHA-256:
`5436e3569ad7c8ca25616ea041dcf471cf2eb87c1e6dd76f3092c61dbaeaf91d`.

## Dispatch issue found after these runs

Launch counts showed 28 fused add/norm launches per step; the layer-boundary
path should execute 56. PEFT 0.20 forwards `output_attentions=None`,
`output_hidden_states=None`, and `return_dict=None`; the blanket `kwargs` guard
rejected these default requests. The earlier varlen and attention-checkpoint
flags were therefore not exercising their intended model-level path.

The fix accepts only supported default requests, retaining original arguments
for fallback. A real PEFT/Qwen2 test asserts packed 2-D input to the first query
projection and compares loss and adapter gradients. The fix is awaiting GPU
measurement; no speed gain is claimed yet.

## Storage semantics and sources

The MLP codec calculates its forward from original values and approximates only
gate/up tensors saved for backward. Int8 uses block scales; int4 uses our
independent symmetric nonlinear codebook, packing two codes per byte. Frozen
weights and the task stay unchanged. Eval, CPU, trainable MLPs, empty inputs and
unsupported dtypes retain the original path.

Saved-activation compression was informed by the general HyC-LoRA idea and
[Team La's public report](https://github.com/Rufus602/soup-dauren-experiments).
Our codec kernels and tests are independent implementations. Parallel CE did
not improve measured training and is not selected. Startup flags mean
`SOUP_X_FPRINT=1` and `SOUP_X_MERGE_VARIANTS=1`; exact flags are in the JSON files.

## Evidence and reproduction

Original logs, summaries and results are kept beside this report with prefixes
`stage2-`, `qualification-v3-`, and `stage3-`. Runners live in `scripts/`:
`performance_stage2.py`, `qualify_candidates.py`, `performance_stage3.py`, and
`performance_dispatch.py`. The full qualifier uses the unchanged harness,
seed 2026, two repeats per side, and all ten checks.

Saved versions of the private experimental Kaggle notebook:
[stage 2](https://www.kaggle.com/code/yujio2x/mononyx-t4-optimization-screen?scriptVersionId=364261629),
[qualification](https://www.kaggle.com/code/yujio2x/mononyx-t4-optimization-screen?scriptVersionId=364266515),
[stage 3](https://www.kaggle.com/code/yujio2x/mononyx-t4-optimization-screen?scriptVersionId=364269452),
[dispatch screen](https://www.kaggle.com/code/yujio2x/mononyx-t4-optimization-screen?scriptVersionId=364271688).
