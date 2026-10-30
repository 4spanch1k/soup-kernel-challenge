# Pending T4 diagnostics (not run: Colab GPU quota was exhausted on 2026-10-31)

Branch `x/diag` (scripts only, `solution/` identical to main 6f3a11e). Run on any T4 (Kaggle notebook of the teammate, or another Colab after the quota resets),
after `bash scripts/kaggle_1_tests.sh` (packages + data) in a clone of the repo:

```bash
git fetch -q origin && git checkout -q origin/x/diag && cp scripts/t4_diag.sh /tmp/ && bash /tmp/t4_diag.sh 2>&1
```
It prints (1) a per-step table for the public rows and for an all-512-token file built only from public rows: valid tokens, memory after the model forward, peak allocated/reserved;
this answers where the unexplained +1,520 MiB peak (6,961 vs 5,441 MiB) on the all-512 file comes from; (2) the head dX GEMM benchmark (suspect: only 12 output tiles for 40 SMs, up to about 1.2 s).
Decision rule: see docs/HANDOFF_FINAL_NOTE.md on branch x/causal-attn; the memory package (attention checkpoint + exact residual, about -670 MiB) is only worth coding if this diagnostics confirms the peak is made of saved activations.

Already measured earlier today (Colab T4, solution-only): main vs x/causal-attn, public rows: 59.93/59.87 s and 60.68/60.08 s, peak 5,441 MiB both: no gain (about 0.3 s mean, inside noise).
