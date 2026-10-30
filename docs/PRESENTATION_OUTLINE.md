# Presentation outline

## Slide 1 — Fixed task, unchanged training setup

- Soup CLI 0.77.0 with Qwen2.5-1.5B-Instruct and NF4 weights.
- LoRA r=16 on `q_proj` and `v_proj`.
- Batch 4, up to 512 tokens and 64 steps, fp16 on a Tesla T4.
- Model, data, batches, optimizer, schedule, LoRA settings, and Soup switches stayed fixed.

**Speaker note:** “We optimized implementation details inside the fixed training task. These measurements are from Google Colab on a Tesla T4.”

**Visual:** A compact task card with the model, training setup, and GPU; label the environment “Google Colab.”

## Slide 2 — What the baseline spends time and memory on

- The shape-recording profile showed an fp32 residual stream.
- `Linear4bit` returns its result in the input dtype, so fp16 results can be cast back to fp32.
- Casts and adds used about 17% of GPU time in that profile.
- The output head created full vocabulary logits and separate loss and metric passes.

**Speaker note:** “The cast path and the output head gave us two clear places to reduce repeated work and intermediate storage.”

**Visual:** A profile bar chart described in words: show the casts-and-adds bar labeled “about 17%.” Beside it, draw an unscaled block for full vocabulary logits and their extra passes; do not assign that block a numeric share.

## Slide 3 — Operations we replaced

- Chunked output-head loss and metrics replace full-logit loss and metric passes.
- Fused RMSNorm replaces the sequence of normalization operations.
- Fused RoPE rotates Q and K without the separate slice, rotate, multiply, and add operations.
- Triton NF4 dequantization handles supported nested NF4 weights.
- Fused add-RMSNorm combines the post-attention residual addition and normalization.

**Speaker note:** “Each kernel targets an operation already present in the fixed run. We wrote the Triton code in this repository.”

**Visual:** A model-flow diagram. Highlight the output head, RMSNorm, RoPE on Q/K, NF4 dequantization before linear operations, and post-attention add plus norm.

## Slide 4 — Why the fp32 round trip matters

- `prepare_model_for_kbit_training` keeps the embedding in fp32.
- The residual stream therefore stays fp32 through the relevant path.
- Bitsandbytes `Linear4bit` returns output in the dtype of its input.
- Repeated fp16-to-fp32 casts enlarged saved activations and added work.
- RMSNorm writes normalized activations directly in the autocast dtype while the residual remains fp32.

**Speaker note:** “We preserve the required fp32 residual values. The normalized values are written in the dtype their consumers use, avoiding the round trip through fp32.”

**Visual:** A two-lane dtype diagram: fp32 residual stream remains fp32; normalized activation is written directly in the autocast dtype before the linear consumer. Mark the removed cast-back path with a simple crossed arrow.

## Slide 5 — Chunk the output head, preserve its outputs

- The head projects 256 token rows at a time.
- Online log-sum-exp computes cross-entropy, entropy, and argmax information per chunk.
- The chunk is reused for the loss gradient; hidden gradients are computed chunk by chunk.
- Padding rows are dropped, and only labelled rows need the loss gradient.

**Speaker note:** “The kernel avoids holding the full vocabulary-logit tensor while still supplying the loss and statistics Soup records.”

**Visual:** A flow diagram from hidden states to token chunks, then to loss, entropy, accuracy, and hidden gradient. Show the temporary logits chunk being reused and released.

## Slide 6 — Base versus our measured run

- Whole `soup train`: 77.5 seconds base, 65.8 seconds ours; 1.176x speedup in this pair.
- Peak GPU memory: 14,345 MiB base, 6,321 MiB ours; 2.269x lower.
- Eval loss: 2.04966 base, 2.04948 ours; -0.009% versus base.
- These are one base run and one solution run on a Colab Tesla T4.

**Speaker note:** “This pair improved time and peak memory, while eval loss stayed very close. It is a single measured pair, not a stable average.”

**Visual:** A base-versus-ours table and paired bar charts for elapsed seconds and peak GPU memory. Put exact values on the bars; use the README result table for eval loss.

## Slide 7 — Harness checks and regression evidence

- All 10 harness checks passed on the Colab T4 run.
- Check 4 recorded a 6.4% combined kernel share, with about 1.4 percentage points of margin.
- Check 6 loss-curve gap was 0.01% on the first step and 0.04% on average.
- Check 8 measured time and memory at 0.850x and 0.441x of base.
- Soup’s 79 regression tests and all 37 solution tests passed on the T4.

**Speaker note:** “The full check table passed. Check 4 has limited margin, so its result deserves particular attention in another environment.”

**Visual:** The recorded `check_rules.py` table with all statuses visible, plus callouts for checks 4, 6, and 8. Add a separate test-output strip for the Soup and solution suites.

## Slide 8 — Limits and honest status

- The measurements are from Colab, not Kaggle; phone verification currently blocks the team’s Kaggle GPU and Internet access.
- There is one measured run per side; hidden jury rows and two runs per side can move the result.
- Check 4 passed by about 1.4 percentage points; the margin is limited.
- MLP recomputation is off by default and was not measured.
- RMSNorm consumer gradients are summed in fp16 under autocast; Soup’s GradScaler is enabled, and the mean loss-curve gap was 0.04%.

**Speaker note:** “These limits define what we can claim today. We do not claim a Kaggle result or a measured recomputation result.”

**Visual:** A status card with “Measured: Colab T4,” “Not measured: Kaggle,” “One run per side,” and “Recomputation off.”

## Slide 9 — Next steps

- Complete the base-versus-solution run on a Kaggle T4 when account access allows it.
- Repeat measurements to assess run-to-run variation.
- Recheck the check 4 margin in the jury environment.
- Measure optional MLP recomputation separately before making any performance claim.
- Complete and share the public notebook and submission links when available.

**Speaker note:** “The next evidence should come from the jury-like Kaggle environment and repeated measurements. Recompute stays an experiment until measured.”

**Visual:** A short sequence: Kaggle run → repeated measurements → check review → optional recomputation experiment.

## Q&A preparation

1. **Is the NF4 kernel really yours?** Yes. `solution/nf4_dequant.py` contains our `@triton.jit` decoder for packed NF4 values and nested scales; it replaces bitsandbytes dequantization only for supported nested NF4 states. We do not import a ready-made kernel library.
2. **Why does eval loss not change?** It changes slightly: 2.04966 for base and 2.04948 for ours, or -0.009% versus base. The measured mean loss-curve gap was 0.04%.
3. **How do you handle fp16 overflow?** The head’s online log-sum-exp uses fp32 values and max subtraction; Soup’s GradScaler remains enabled. We report the observed loss-curve gap and do not claim that overflow is impossible.
4. **Why did you use Colab?** The team’s Kaggle account lacks phone verification required for GPU and Internet access, so the recorded T4 measurement is from Colab.
5. **Does the recompute switch break the rules?** `solution/recompute.py` is our own optional MLP checkpointing, not Soup’s forbidden `gradient_checkpointing` switch. It is off by default and was not measured.
6. **Will the speedup hold on Kaggle?** We do not know yet. The 1.176x figure is from one base/solution pair on Colab; the README reports an observed range of roughly 1.07x–1.18x across development and base-run variation.
7. **Do your kernels account for enough training time?** The measured combined share was 6.4%; check 4 passed by about 1.4 percentage points, so the margin is limited.
8. **Did you use Liger-Kernel, Cut Cross-Entropy, or Soup’s `stream_loss_chunk`?** Their ideas or outputs informed the implementation, but those libraries and the Soup switch are not imported or enabled. The kernels are written in this repository.
