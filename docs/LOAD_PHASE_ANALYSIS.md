# Load phase analysis

## Scope and decision

This review follows the fixed `task/task.yaml` path in `harness/_train.py`, Soup CLI 0.77.0's
`SFTTrainerWrapper._setup_transformers`, Transformers 5.16.1, and bitsandbytes 0.50.2. The local
machine has no CUDA GPU, so all T4 timings below are either reported by the user / handoff or
explicitly labelled estimates. The new timing script is for the T4 owner to measure the real stages.

I found no load-path patch that is both value-preserving and likely to save at least 0.5 seconds.
Accordingly, this change adds no `solution/load_phase.py` and does not alter `solution/__init__.py`.
That avoids moving work into a different phase, bypassing Soup's loader, or maintaining a second
implementation of PEFT / bitsandbytes behavior without a T4 result.

The wall-time reports are approximate and do not add up consistently: 63 seconds total minus a
35–42 second loop implies 21–28 seconds outside it, while “about 21 seconds” is the low end. The
14-second `from_pretrained` figure is an estimate, not a stage measurement. The handoff separately
records a Colab run at 63.5 seconds wall / 42.1 seconds in the loop; its startup experiment measured
1.5 seconds for `torch` import, 8.2 seconds for `transformers` / `trl` / `peft` / `bitsandbytes`,
and 1.1 seconds for `solution.apply()`. Those are useful prior measurements, not measurements of
this worktree or the requested T4 run.

## Real call path

1. `harness/run_task.py` warms package and model caches in a separate process, then starts
   `harness/_train.py`. The warm-up is outside the measured run; the Triton compiler cache for each
   measured subprocess is isolated.
2. `_train.py` sets the Soup CLI arguments, imports `torch`, creates the CUDA context, imports
   `transformers`, installs the Triton launch watcher, and calls `solution.apply()` for a solution
   run. It then wraps `transformers.Trainer.__init__` to record settings and adds a callback that
   synchronizes CUDA at each step boundary. These imports and setup occur before `soup_cli.cli.run()`.
3. `soup_cli.cli.run()` enters the `soup train` command. The command reads the task config, hashes
   the training data, loads and validates the raw JSONL rows, starts the experiment tracker, creates
   the trainer wrapper, then calls `setup(dataset)`.
4. `setup()` imports `TrainingArguments` / `SFTTrainer`, seeds the run, and calls
   `_setup_transformers()`. That function imports PEFT; loads the tokenizer; builds the quantization
   config; resolves dtype and device map; and calls `AutoModelForCausalLM.from_pretrained()`.
5. After loading, Soup applies vocabulary expansion and model checks, calls
   `prepare_model_for_kbit_training()` for 4-bit training, resolves the `q_proj` / `v_proj` targets,
   builds the LoRA config, applies Soup's configured PEFT wiring, and calls `get_peft_model()`.
6. Soup formats and tokenizes the 256 training rows (up to 512 tokens each), creates
   `TrainingArguments` / `SFTConfig`, and initializes the `SFTTrainer` (the rewind trainer is used
   for this task). The trainer builds its data loader and lazily builds optimizer state when training
   begins.
7. `trainer.train()` starts the 64 steps. The first forward also compiles kernels that have no
   compiled cache. The handoff's separate startup experiment measured 4.7 seconds to compile all
   solution kernels with an empty Triton cache; this is historical Colab data, not a fresh T4 result.
8. Soup saves the final adapter and tokenizer, closes the rewind log, writes run / experiment
   metadata, and returns through the CLI. The harness waits for the child process to exit before
   stopping its wall timer, so final writes and normal interpreter shutdown are included.

## Cost estimates by stage

For arithmetic, use the model's nominal 1.5 billion parameters and two-byte BF16 checkpoint values:
`1.5e9 × 2 = 3.0 GB` (`2.79 GiB`) of weight payload. The Qwen2.5-1.5B dimensions visible in the
solution tests (hidden 1536, intermediate 8960) imply about 1.33 billion parameters in the seven
linear matrices per layer (`28 × (2 × 1536² + 2 × 512 × 1536 + 3 × 8960 × 1536)`). This is an
architecture estimate; the loaded checkpoint header is authoritative.

| Stage | Bytes / copies / GPU work | Redundant for this task? | Safe speed option and decision |
|---|---|---|---|
| Python imports and CUDA context | The handoff measured 1.5 s `torch` import + CUDA-independent import work, 8.2 s for the other four ML packages, and 1.1 s for `solution.apply()`. No model-weight bytes. The CUDA context is initialized before Soup runs. | No. Soup, PEFT, Transformers and bitsandbytes are required by the fixed training path. The solution imports its active kernels. | Imports are already a large shared cost. Removing libraries or deferring imports does not remove work needed before training, and moving imports later only shifts wall time. No patch. |
| Tokenizer load | Reads tokenizer config / vocabulary / special-token files from the local Hub cache; small metadata and vocabulary payload compared with the model weights. No CUDA kernels. | No. The same tokenizer formats and masks the required data and is saved at the end. | A local tokenizer cache is already used by the normal run. Skipping load or using a prebuilt token cache would change the allowed path. No patch. |
| Quantization config and model config | Soup builds `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=get_compute_dtype(), bnb_4bit_use_double_quant=tcfg.double_quant_on)`. On T4 the compute dtype resolves to fp16. Config / tokenizer metadata are tiny; no meaningful GPU work. | No. It selects the fixed NF4 task settings. | Construction is small. Disabling checks or warnings does not remove meaningful work and could hide incompatibility. No patch. |
| `device_map`, meta model construction, and checkpoint discovery | Soup's `resolve_device_map('cuda')` returns `"auto"` for a single process. Transformers resolves the map, discovers safetensors shards, and creates the model under `torch.device('meta')`. Model object / parameter metadata are small relative to weights. | The `"auto"` map inference is unnecessary in principle for one known GPU, but is a small CPU traversal over modules. | Replacing it with a fixed root-to-GPU map may save map-planning work, but no source evidence suggests a 0.5 s saving. It can change placement behavior if the hardware / launch differs. Not implemented. |
| `low_cpu_mem_usage` | Transformers 5.16.1 removes `low_cpu_mem_usage` from `from_pretrained` kwargs. Its `get_init_context()` already puts the model skeleton on `meta` for this quantized path. The safetensors loader stores mmap-backed slices, then materializes / transfers parameters as they are assigned. | There is no extra low-memory switch to remove in this version. | Passing `low_cpu_mem_usage=True` or `False` is ignored by the inspected source; it cannot speed this path. No patch. |
| Safetensors read and BF16 → fp16 / NF4 load | Up to about 3.0 GB of checkpoint values are read logically. Soup requests `torch_dtype=torch.float16` on T4. Transformers materializes each mmap-backed safetensors slice and calls `tensor.to(device=param_device, dtype=dtype)` before the BNB conversion; for the linear weights, this is the per-tensor BF16 → fp16 / device-placement boundary. The source does not show whether PyTorch's combined CPU-to-CUDA `.to()` performs the dtype conversion on CPU or as part of its CUDA copy path, and this needs profiling to split further. This is not a second whole-model copy. For about 1.33B linear values, the input is about 2.66 GB, packed NF4 output about 0.67 GB, and first-level scales about `1.33B / 64 × 4 ≈ 83 MB` of temporary fp32 values before double quantization. Actual physical disk reads depend on the warmed OS / Hub cache. | No. Loading and quantizing those same checkpoint values is required. | CPU conversion would add a full read + write before transfer. GPU conversion still reads each value and cannot skip NF4 quantization. Pinned loading may help only if transfer is the bottleneck; source uses per-tensor loading and there is no T4 measurement showing a ≥0.5 s gain. No patch. |
| bitsandbytes NF4 + double quantization | Transformers disables its materialization thread pool for on-the-fly quantization, so it does not load full-precision tensors in background workers ahead of BNB. `Params4bit._quantize()` makes the tensor contiguous / places it on the target, then calls `quantize_4bit`. Per quantized tensor, code performs a primary 4-bit quantize op; with double quant it computes the mean of the already-produced block absmax values and calls `quantize_blockwise` on those scales. At roughly 196 Qwen linear matrices this is about 196 primary ops + 196 scale reductions + 196 blockwise scale ops; exact low-level launch count is backend-dependent. The scale quantization works on about 20.8M scales, not on the 1.33B original values again; it writes about 20.8 MB of 8-bit scales plus ~0.33 MB second-level fp32 absmax. The source has no quantized-weight CPU round trip. | No. NF4 values and double-quant scales are the requested representation. The second pass is only for scales and is much smaller than the weight pass. | Fusing or replacing these kernels would need bit-exact output and quant-state tests against BNB and a T4 win. No such kernel is already in the repository; a Python patch alone cannot safely eliminate this work. No patch. |
| `tie_weights` and generation config | Transformers suppresses tying while creating the meta skeleton, then calls `model.tie_weights()` after loading. Tying adjusts parameter references; it does not copy a whole table. `PreTrainedModel` creates a `GenerationConfig` from `model.config`; the inspected load path does not call `GenerationConfig.from_pretrained()`. | No expensive duplicate weight copy is evident. | Removing tie logic risks aliasing / tied-head behavior. Generation-config creation is CPU object work. No patch. |
| PEFT k-bit preparation | `prepare_model_for_kbit_training()` freezes parameters, may enable input grads / gradient checkpointing per its arguments, and casts non-quantized fp16/bf16 parameters to fp32. The tied 233,373,696-value embedding is about 466.7 MB fp16 and 933.5 MB fp32. The solution's existing `compact_tied_embedding()` later verifies fp16 exact representability, restores the fp16 table, ties it to the head and emits fp32 embedding outputs. Upcast + later downcast moves about 2.8 GB of aggregate GPU read/write traffic (466 MB read + 933 MB write, then 933 MB read + 466 MB write), plus two conversion launches; at even 50 GB/s this is ~56 ms of bandwidth. Other non-quantized parameters are small. | The upcast/downcast is duplicate traffic for the solution's exact-fp16 embedding, but removing it needs a replacement PEFT implementation or a version-specific patch. | The upper-bound arithmetic is well below 0.5 s. A custom PEFT copy risks changing hook, gradient-checkpointing or dtype behavior. The current later compaction is guarded by an exact-value check. No patch. |
| LoRA setup | The fixed rank-16 adapters attach to q/v in 28 layers (56 modules). Estimated trainable count is `28 × 16 × (1536 + 1536 + 1536 + 512) ≈ 2.29M` values: ~9.2 MB at fp32. Python module traversal and small allocations / initializers; no large checkpoint copy. | No. These are the requested trainable parameters. | Existing Soup target resolution and `get_peft_model()` are necessary. Replacing them or changing LoRA initialization is not allowed. No patch. |
| Dataset tokenization / trainer init | At most `256 × 512 = 131,072` token positions, with CPU formatting/tokenization and small Arrow / Python structures. No GPU kernels expected before the first batch. Trainer / dataloader initialization is mostly Python; optimizer state is lazy and only for ~2.29M LoRA values. | No. The task must use these rows, order, tokenizer and trainer semantics. | Tokenization caches would be prebuilt data and risk a different encoding. Trainer shortcuts could change metrics or batch handling. No patch. |
| First step, compile, and syncs | The first step runs the full training path. Fresh Triton compilation is included; the handoff's 4.7 s cold compile is only a prior measurement. `_train.py` synchronizes at callback step boundaries, and the loop timer includes those synchronizations. | No. First-step compile is part of the measured run; syncs make timing boundaries accurate. | Background compile / warm-up and prebuilt caches are expressly disallowed. Removing harness syncs changes the measurement harness. No patch. |
| `torch.compile` / Inductor | The fixed task does not enable `TrainingArguments.torch_compile`; there is no `torch.compile()` call in the inspected path. No generated Inductor cache is needed by this solution. | No compilation is required by the task. | Enabling compile changes execution and import / compile cost, and cached binaries are prohibited. No patch. |
| Adapter save, Soup metadata, exit | No periodic checkpoint is due (`save_steps=1000`, 64 steps). Final adapter is on the order of 9 MB if 2.29M fp32 parameters; JSON/config, tokenizer files, up to 256 rewind rows, tracker DB, hashes and run provenance add file operations. Plain `soup train` does not render a model card; that is a separate `soup card` / push flow. Exact file set and fsync / exit-hook times need T4 timing. | Saving these outputs is required; only intermediate checkpoint saving is skipped by the fixed schedule. | Adapter bytes are small; skipping rewind, experiment records, hashes or tokenizer save would change Soup outputs. No justified patch. |

## Candidate conclusions

- **Safetensors conversion / pinned loading:** the current path is already streamed by parameter through mmap slices, target dtype and quantizer placement. A CPU fp16 staging copy adds bytes; pinned transfer is unproven as the bottleneck. Measure first.
- **`device_map` / `low_cpu_mem_usage`:** `"auto"` map inference is a possible small cleanup, but the 5.16.1 meta initialization is already in effect and `low_cpu_mem_usage` is discarded. No 0.5-second evidence.
- **NF4 double quant:** the mean + blockwise pass is over the block-scale array. It does not repeat the full weight absmax pass. Changing it threatens quant-state identity for little plausible time.
- **PEFT upcast:** real redundant copy with this solution's exact fp16 compaction, but a conservative bandwidth estimate puts it under 0.1 seconds. A custom replacement for PEFT's setup is too risky for that ceiling.
- **Tokenizer, configs, tying, dataset, trainer, saving:** required or small. No safe stage skip with a plausible >0.5-second reduction.
- **Imports / compilation / exit:** imports are a shared fixed dependency cost; solution imports are required. Compile hiding and caches are prohibited. Exit time is not isolated by the supplied source and cannot be safely changed.

## How to measure on the T4

The new script uses Soup / Transformers / PEFT's real setup calls in one process, with no timing
monkeypatch and no `solution.apply()`. It prints import, data hash / read, tokenizer,
`from_pretrained`, PEFT preparation, and LoRA timings. It stops after adapter creation; it does not
tokenize, initialize the trainer, train, save outputs or modify the task. Use the full harness
command to measure trainer initialization, first step, final save and exit as one real run.

```bash
PYTHONPATH=. python scripts/load_phase_timeline.py
python -m pytest solution -q
bash scripts/kaggle_3_full.sh
```

Do not interpret the diagnostic script's per-call totals as an end-to-end wall timer: it excludes
dataset mapping, trainer construction, training, save, and process exit. Compare those stage values
with the paired wall / loop values printed by the full harness.
