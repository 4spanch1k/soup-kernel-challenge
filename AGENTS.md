# Agent instructions (Codex, Claude Code, any other agent)

Project: Soup Kernel Challenge (Decentrathon 6.0). Read `README.md` for the task and
`docs/PLAN.md` for our analysis and plan before changing anything.

## Hard rules (breaking one disqualifies the team)
- Edit only `solution/`, `docs/`, `scripts/`, `notebook/`, `AGENTS.md`, `CLAUDE.md`, `README.md`,
  `SUBMISSION.md`. Only `solution/` is copied into the jury's run; the rest is our tooling and docs.
  Never edit `harness/`, `tests/`, `task/`; the jury uses its own copy.
- Do not change the task: model, data, batch order, steps, LoRA settings (dropout too), optimizer,
  schedule, Soup switches (`unsloth`, `use_liger`, `use_cut_ce`, `stream_loss_chunk`,
  `gradient_checkpointing`, `activation_offloading`, `fast_lora`, `packing`, `padding_free`).
- No ready-made kernel libraries (unsloth, liger-kernel, xformers, flash-attn, torchao, full list in
  `harness/common.py`). No `pip install` during a run.
- Our kernels must be our own `@triton.jit` code written in `solution/`. Do not rename or rewrite
  `solution/swiglu.py` (the example does not count). Name ideas taken from elsewhere in the README.
- Soup outputs must keep their values: steps, tokens per step, `entropy`, `mean_token_accuracy`,
  `rewind.jsonl` row losses (within 3%), eval loss at most 0.5% above base.
- T4 means fp16, no bf16. No CUDA graphs (the harness cannot see the kernels).

## Working together
- One owner per file. Ownership is in the table below; ask before touching another's file.
- Work on your own branch (`git switch -c <owner>/<topic>`), small commits, rebase on `main` before
  merging. `solution/__init__.py` is shared: only add one line to `apply()` per kernel.
- Every kernel gets a test in `solution/test_<name>.py` against plain PyTorch, forward and
  backward, fp16 and fp32, odd shapes. The dev machine (Mac) has no GPU: tests skip there, the
  real run is on Kaggle T4. Never claim a kernel works without a GPU run; say "not run".
- Patch narrowly and guard (`isinstance(self.model, torch.nn.Module)`): Soup's 79 tests run with
  `apply()` active: `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q`.
- Fixed Triton configs, no wide autotune: compilation time counts in the measured wall time.

| Owner | Files |
|---|---|
| A | `solution/ce_loss.py`, `solution/head_patch.py`, `solution/test_ce_loss.py` |
| B | `solution/rmsnorm.py`, `solution/rope.py` and their tests |
| C | `solution/recompute.py`, padding trim, memory measurements |
| D | `README.md`, `SUBMISSION.md`, Kaggle notebook, video, slides |

## Commands
```bash
python -m pytest solution -q                         # kernel tests (needs a CUDA GPU)
python harness/run_task.py --only solution --out runs_try   # quick solution-only run
python harness/run_task.py && python harness/check_rules.py  # full measurement + the 10 checks
```
