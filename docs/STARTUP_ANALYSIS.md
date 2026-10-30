# Startup time analysis

## What the harness measures

`harness/run_task.py` times each `_train.py` subprocess from process start through exit. Its
separate warm-up subprocess imports the libraries and downloads the model files first; that
subprocess's time is excluded. Every measured subprocess still repeats its Python imports and loads
the model from the local cache. Each measured process gets a fresh `TRITON_CACHE_DIR` and
`TORCHINDUCTOR_CACHE_DIR`.

`harness/_train.py` imports PyTorch, initializes CUDA, imports Transformers, installs its Triton
watcher, then imports `solution` and calls `solution.apply()` before importing/running
`soup_cli.cli`. Soup reads the configured dataset, loads the tokenizer and model, quantizes NF4
weights, prepares k-bit training, builds LoRA and the trainer, maps/tokenizes rows, trains, saves
the adapter and tokenizer, and exits.

One timing distinction matters: Triton JIT compilation and the first-step CUDA warm-up happen
during the first training step. They are included in the callback's `train_loop_s`, even though
they are excluded from `_train.py`'s steady-state own-kernel timer, which starts at step index 2.
For the reported 63.5-second run and roughly 35-second loop, the remaining roughly 28.5 seconds
are imports/setup/model and data loading plus final save/exit. The 35 seconds still include first
step compilation.

## Approximate wall-time budget

The only measured totals available here are the README's 63.5-second complete solution run and the
owner's recent estimate of about 35 seconds in the training loop. The phase split below is an
engineering estimate, not a profile; model loading and quantization are the largest unknown.

| Phase | Estimate | Evidence and confidence |
|---|---:|---|
| `_train.py`, Python/PyTorch/Transformers/Triton watcher imports and `solution.apply()` | 4–8 s (point budget 5.5 s) | Order is explicit in `_train.py`; duration is guessed. |
| Soup CLI, local dataset read and format/tokenization maps | 1–4 s (point budget 2 s) | Calls are visible in Soup's `cli.py`, `commands/train.py`, and `trainer/sft.py`; map duration is guessed. |
| Tokenizer load | 0.5–2 s (point budget 1 s) | `AutoTokenizer.from_pretrained` is explicit; duration is guessed. |
| Qwen load and NF4 quantization | 12–18 s (point budget 14 s) | `AutoModelForCausalLM.from_pretrained` uses fp16 plus the configured NF4 quantizer; GPU I/O and conversion duration are guessed. |
| K-bit preparation, LoRA and trainer construction | 2–5 s (point budget 3 s) | PEFT and `SFTTrainer` construction are explicit; duration is guessed. |
| First step, including Triton JIT and CUDA library warm-up | 5–12 s | Included in the 35-second callback loop; split from the other steps is guessed. |
| Remaining training steps | 23–30 s | Inferred by subtraction from the approximately 35-second loop. |
| Checkpoint/adapter/tokenizer save, Soup return and process exit | 2–5 s (point budget 3 s) | Save calls are explicit in `sft.py`; duration is guessed. |
| **Total** | **about 63.5 s** | Point budget totals 63.5 s; all phase values except whole run and callback loop are estimates. Ranges overlap. |

`task/prepare_data.py` and `run_task.py`'s warm-up/download are outside the per-side timed subprocess.
The dataset is still read and tokenized inside the timed Soup process.

## Kernel compilation count and changes

The Triton launch dimensions and constexprs in `solution/` show that ordinary row-count changes do
not create variants in RMSNorm or add+RMSNorm: their row count is the launch grid, while hidden
width and block width remain fixed. The output-head row count is also a grid dimension; its two
`WRITE_GRAD` constexpr values do make separate variants. Add+RMSNorm backward has two necessary
`HAS_DSUM` variants.

RoPE used to pass `seq` as a scalar and derive batch/position with integer divide and remainder.
The training batches use different sequence lengths (`T=200`, `311`, and `512`). Triton can compile
different specializations when that scalar's alignment class changes. `rope_kernel` now receives a
three-dimensional grid `(seq, heads, batch)` and reads those indices directly. This removes the
sequence scalar, its divide/remainder, and the sequence-dependent specialization while preserving
the same per-element arithmetic. The existing `solution/test_rmsnorm_rope.py` compares forward and
backward against Qwen2 on a non-power-of-two sequence; the T4 run still needs to confirm compilation
and timing.

Expected source-level variants before that change were about 12 for the solution's own kernels:

| Kernel | Expected variants | Why |
|---|---:|---|
| `rmsnorm_fwd_kernel` / `rmsnorm_bwd_kernel` | 1 + 1 | Fixed hidden width and dtype; rows are grid size. |
| `add_rmsnorm_fwd_kernel` | 1 | Fixed width and dtype. |
| `add_rmsnorm_bwd_kernel` | 2 | `HAS_DSUM=True` and `False`. |
| `rope_kernel` | up to 4 before, 2 after | Forward/backward constexprs times up to two sequence alignment classes; the new grid removes the sequence class. |
| `head_stats_kernel` | 2 | `WRITE_GRAD=True` and `False`. |
| `nf4_dequant_kernel` | 1 | Fixed block and nested-quantization settings. |

This is a code-derived upper estimate, not an observed Triton cache inventory. The profiler script
below reports cold/warm launch deltas for the actual T4 compiler. `solution.apply()` installs the
SwiGLU example patch, but the profiler does not construct a model or launch `example_*`; the harness
also excludes that example from the custom-kernel share.

## Savings and constraints

| Candidate | Exact place | Assessment |
|---|---|---|
| Remove sequence-dependent RoPE variants and integer divide/remainder | `solution/rope.py`, `rope_kernel` program IDs and `_launch` grid | Implemented. It should remove up to two JIT variants (one per direction) and a per-program integer divide/remainder. Expected compile saving is roughly 2–6 s if both variants were compiled and each cost 1–3 s; confidence is medium for fewer compiles, low for wall-time savings until the T4 profile. |
| Avoid a second CUDA synchronization per embedding scan chunk | `solution/embed16.py`, `_is_exact_fp16` | Implemented by reducing both boolean checks before one `.item()`. It preserves the original finite-and-exact-fp16 predicate. At the current chunk size this removes 14 synchronization points for the full embedding. Expected saving is roughly 0.05–0.5 s; confidence is low without a GPU measurement. |
| Merge `WRITE_GRAD` or `HAS_DSUM` variants | `solution/ce_loss.py:head_stats_kernel`, `solution/add_rmsnorm.py:add_rmsnorm_bwd_kernel` | Not changed. A runtime branch may reduce cold compilation but run on every invocation; benchmark first because these kernels execute repeatedly and are part of the checked kernel share. |
| Mark RoPE `seq` as `do_not_specialize` | `solution/rope.py:rope_kernel` decorator | Superseded by removing `seq` from the kernel arguments and indexing the grid directly; this avoids dynamic division as well as the specialization. |
| Skip or parallelize dataset formatting/tokenization | Soup `trainer/sft.py:_map_text_sft_rows` and its `Dataset.map` call | Potential startup work, but batching or multiprocessing can change tokenizer call behavior, row/error ordering, or dataset fingerprints. Only consider after verifying byte-for-byte `input_ids`, labels, row order, and rewind rows. |
| Avoid model dtype conversions in k-bit preparation | Soup `trainer/sft.py` around `from_pretrained` and `prepare_model_for_kbit_training`; PEFT implementation | Potentially expensive, but changes model parameter dtypes/storage used by embedding, head, and LoRA preparation. The existing fp16 tied-embedding compaction checks exact representability; do not bypass it without T4 output and quality checks. |
| Remove tokenizer save or trainer output work | Soup `trainer/sft.py` final save | Not viable: the harness consumes the adapter, trainer state and rewind output. |
| Reduce `soup_cli.cli` import cost | Soup's top-level imports in `soup_cli/cli.py` | The CLI imports command modules before `run()` dispatch. `apply()` cannot avoid that import after Soup starts, and importing it early only moves the same cost. A local CLI rewrite is outside this task's safe patch boundary. |

The base and solution both pay for package imports, dataset processing, model loading and trainer
setup. Pure reductions there can lower total wall time, but changes to data order, model values,
LoRA settings or Soup output are not acceptable. Patching Soup/Transformers/PEFT code is technically
possible from `solution.apply()`, but only when it preserves those invariants; no such patch was
clear enough to make without the T4 output checks.

`solution.apply()` imports Torch/Triton code and patches Qwen2, TRL and bitsandbytes hooks. Those
libraries are needed later by Soup; `_train.py`'s watcher has already imported Triton and the
runner has already imported Torch and Transformers. This does not add another copy of those heavy
imports, and the apply-time Qwen2/TRL/bitsandbytes imports are reused by the later training setup.
The profiler times those library imports as a separate group and `solution` plus `apply()` after
them, so its apply number is incremental and lower than a hypothetical all-in-one import phase.

## T4 measurement

Run from the repository root:

```bash
python scripts/startup_breakdown.py
python -m pytest solution -q
bash scripts/kaggle_3_full.sh
```

`startup_breakdown.py` creates a new temporary `TRITON_CACHE_DIR` before importing PyTorch or
Triton, times the requested Python imports and `solution.apply()`, initializes CUDA, then calls
each custom JIT kernel twice with realistic dimensions. Each line reports first call, second call,
and their difference. That difference estimates JIT cost; it also includes first-use launch effects
and should be treated as an estimate rather than a compiler trace. The full harness remains the
authority for startup wall time, Soup outputs and score.
