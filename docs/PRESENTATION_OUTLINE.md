# Presentation outline

Final numbers come from one paired run per environment on source commit `d881159`: an interactive
Kaggle Tesla T4 session and a Colab Tesla T4 session. Do not compare raw seconds across platforms;
compare base and solution within each session. The public Kaggle saved version 1 remains on pre-fix
commit `b587c5c` and is historical evidence. Values are taken from `README.md`, `SUBMISSION.md`,
and current-run screenshots in `docs/kaggle/screenshots/`.

## Slide 1 — Fixed task, unchanged training setup

- Soup CLI 0.77.0 with Qwen2.5-1.5B-Instruct and NF4 weights.
- LoRA r=16 on `q_proj` and `v_proj`.
- Batch 4, up to 512 tokens and 64 steps, fp16 on a Tesla T4.
- Model, data, batches, optimizer, schedule, LoRA settings, and Soup switches stayed fixed.

**Speaker note:** “We optimized implementation details inside the fixed training task. The final code was measured on T4s in Colab and an interactive Kaggle session; the public saved Kaggle notebook still shows the earlier code.”

**Visual:** A compact task card with the model, training setup, and GPU; label the environments “Kaggle T4” and “Colab T4”.

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
- Triton NF4 dequantization handles supported nested NF4 weights in the matmul forward and in backward.
- Fused add-RMSNorm combines each residual addition with the following normalization, including layer boundaries.

**Speaker note:** “Each kernel targets an operation already present in the fixed run. We wrote the Triton code in this repository; the starter kit's SwiGLU example is not counted.”

**Visual:** A model-flow diagram. Highlight the output head, RMSNorm, RoPE on Q/K, NF4 dequantization before linear operations, and the add plus norm points.

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
- The idea follows Milakov and Gimelshein (2018); Liger-Kernel and Cut Cross-Entropy informed the approach but are not imported.

**Speaker note:** “The kernel avoids holding the full vocabulary-logit tensor while still supplying the loss and statistics Soup records.”

**Visual:** A flow diagram from hidden states to token chunks, then to loss, entropy, accuracy, and hidden gradient. Show the temporary logits chunk being reused and released.

## Slide 6 — Final paired T4 results (Kaggle and Colab)

- Kaggle T4, current source `d881159`: 98.1 s base, 58.5 s ours; 1.677x; 14,345 → 4,591 MiB; eval loss +0.009%.
- Colab T4, same source: 77.3 s base, 54.1 s ours; 1.428x; 14,345 → 4,591 MiB; eval loss +0.010%.
- Each figure is one base/solution pair in a single session; cross-platform seconds are not comparable.
- Public saved Kaggle version 1 (`b587c5c`) is an earlier run: 87.77 → 62.10 s, 14,345 → 5,441 MiB, eval loss −0.006%.

**Speaker note:** “Both paired runs passed the harness. Each is a single measurement, not a stable average. Compare each solution only with its same-session base; do not compare raw seconds between Kaggle and Colab.”

**Visual:** Two equal cards for Kaggle and Colab with base/solution time, speedup, memory, eval-loss delta and checks. Add a note that the public saved Kaggle notebook is pre-fix; its historical figures are documented in `README.md`.

## Slide 7 — Current Kaggle output and harness evidence

- Show the current interactive Kaggle T4 result screenshot: 98.1 → 58.5 s, 14,345 → 4,591 MiB, eval loss +0.009%.
- Show the current harness summary screenshot: all 10 checks passed; check 8 time x0.596 and memory x0.320 of base; 79 Soup regression tests passed.
- These screenshots are from updated code, not the older public saved version 1.
- The public saved Kaggle notebook still shows commit `b587c5c`; disclose that difference.

**Speaker note:** “These are current Kaggle screenshots from the interactive run. The saved public notebook is still the previous version, so the README calls out the distinction.”

**Visual:** Two cropped Kaggle notebook screenshots: final run output and all-checks summary. Keep the older public notebook evidence out of this slide.

## Slide 8 — Limits, rejected ideas and honest status

- Kaggle gives one measured run per side on public data; the jury's hidden rows and two runs per side can move the result.
- Earlier Colab seed repeat (seed 2026, two runs per side, pre-fix code): all 10 checks passed; 80.8 s → 66.2 s (1.221x), 14,339 → 5,419 MiB (2.646x less), eval loss +0.016%, kernels 8.8%.
- MLP recomputation is off: measured at 59.7 s / 6,321 MiB off and 69.3 s / 4,601 MiB on, which is 16% time for 27% memory.
- RMSNorm consumer gradients are summed in fp16 under autocast; Soup's GradScaler is enabled, and the mean loss-curve gap was 0.04%.
- Tried and rejected on the T4: per-sample causal attention (no gain), fused NF4 GEMM in one Triton kernel (10-200 times slower than cuBLAS), LoRA-path fusion (the patch never fired, so there was no gain).

**Speaker note:** “These limits define what we can claim. The rejected ideas are measured, not guessed.”

**Visual:** A status card with “Measured: Kaggle T4, one pair”, “Earlier: Colab T4 and seed repeat”, and “Recomputation off”, next to a small table of rejected ideas.

## Slide 9 — Final submission checks

- The demo video and presentation links are recorded in `SUBMISSION.md`.
- Verify the Kaggle, Pitch, and video links in an anonymous browser session.
- Confirm the judging team's access to the private platform repository before the deadline.
- Keep MLP recomputation disabled: it costs 16% more time for 27% less memory.

**Speaker note:** “The measured code is frozen. The final operational check is that the judges can open the links and the platform repository.”

**Visual:** Show the three public links and the platform access check.

## Q&A preparation

1. **Is the NF4 kernel really yours?** Yes. `solution/nf4_dequant.py` contains our `@triton.jit` decoder for packed NF4 values and nested scales; it replaces bitsandbytes dequantization only for supported nested NF4 states. We do not import a ready-made kernel library.
2. **Why does eval loss not change?** It changes slightly: +0.009% versus base in the Kaggle pair and +0.010% in the Colab pair, both within the 0.5% limit. The public saved version's earlier result was −0.006%.
3. **How do you handle fp16 overflow?** The head's online log-sum-exp uses fp32 values and max subtraction; Soup's GradScaler remains enabled. We report the observed loss-curve gap and do not claim that overflow is impossible.
4. **Why are there Colab numbers as well as Kaggle numbers?** The final source was measured in one paired Colab T4 run and one paired interactive Kaggle T4 run. The saved public Kaggle version 1 still contains pre-fix code; each session is reported separately.
5. **Does the recompute switch break the rules?** `solution/recompute.py` is our own optional MLP checkpointing, not Soup's forbidden `gradient_checkpointing` switch. It is off by default because it cost 16% time for 27% memory.
6. **Will the speedup hold on the jury's run?** We do not know. The current source gave 1.677x on one Kaggle pair and 1.428x on one Colab pair. The saved public Kaggle notebook shows the earlier 1.413x result.
7. **Do your kernels account for enough training time?** The final Kaggle run passed all 10 checks, including the 5% kernel-share requirement. The final Colab run reported 13.1%; the older saved Kaggle run reported 10.8%. Hidden rows may change it.
8. **Did you use Liger-Kernel, Cut Cross-Entropy, or Soup's `stream_loss_chunk`?** Their ideas or outputs informed the implementation, but those libraries and the Soup switch are not imported or enabled. The kernels are written in this repository.
