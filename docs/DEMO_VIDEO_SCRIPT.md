# Demo video script

**Target runtime: 2:45. Hard limit: 3:00.** Use the existing Colab Tesla T4 results. Label them as Colab results; do not present them as a Kaggle run. The full-measurement command is shown as a reproducibility command, while the displayed result is the recorded run described in `README.md`.

| Time | Speaker | On screen | Exact narration |
|---|---|---|---|
| 0:00–0:18 | Speaker 1 | Colab notebook title, then a cropped `nvidia-smi` result showing the GPU name, Tesla T4. | “This is our Soup Kernel Challenge solution. We kept the fixed Soup training task and replaced selected operations with our own Triton kernels. The results I’ll show are from Google Colab on a Tesla T4.” |
| 0:18–0:42 | Speaker 2 | A simple profile bar chart described in words: a highlighted bar for casts and adds labeled “about 17%”; beside it, a large, unscaled block labeled “full vocabulary logits and extra passes.” Add “shape-recording profile” as the chart caption. | “The profile showed two costs. The residual stream stays in fp32, so fp16 linear outputs are cast back and casts plus adds used about 17% of GPU time. The output head also materialized huge vocabulary logits, followed by separate loss and metric work.” |
| 0:42–1:19 | Speaker 3 | Kernel map beside short code clips from `ce_loss.py`, `rmsnorm.py`, `rope.py`, `nf4_dequant.py`, and `add_rmsnorm.py`. | “The output head handles 256 token rows at a time and computes loss, entropy, argmax and gradients without materializing all logits. RMSNorm uses fp32 statistics, saves one reciprocal standard deviation per row, and writes the normalized output in the autocast dtype. RoPE rotates both halves of a head in one forward kernel and applies the inverse rotation in backward. NF4 dequantization decodes packed nibbles and nested scales into the compute dtype. Add-RMSNorm fuses the post-attention residual addition and normalization while preserving the fp32 residual and combining the backward paths.” |
| 1:19–1:34 | Speaker 4 | Terminal command card, then recorded T4 test output. | “The solution test command is `python -m pytest solution -q`. On the T4, all 37 solution tests passed. The full measurement entry point is `bash scripts/kaggle_3_full.sh`; it runs the base and solution measurement, checks, and score.” |
| 1:34–1:59 | Speaker 1 | README results table, enlarged and held long enough to read. | “In this measured pair, whole training took 77.5 seconds for base and 65.8 seconds for ours: a 1.176x speedup. Peak GPU memory went from 14,345 to 6,321 MiB, 2.269x lower. Eval loss was 2.04966 versus 2.04948, a change of minus 0.009 percent.” |
| 1:59–2:14 | Speaker 2 | Recorded `harness/check_rules.py` check table or summary, labeled “Colab run.” Highlight checks 4, 6, 8, and 10; keep all ten statuses visible. | “All ten harness checks passed. Our kernels used 6.4% of training time in check 4. Check 6’s mean loss-curve gap was 0.04%. Check 8 measured time and memory at 0.850x and 0.441x of base. Soup’s 79 regression tests also passed with our patch active.” |
| 2:14–2:34 | Speaker 3 | README Limits section, with “Colab, not Kaggle” and “one run per side” visible. | “These are Colab results, not Kaggle results: our Kaggle account lacks phone verification for GPU and Internet access. We measured one run per side, so hidden rows and the jury’s two runs per side may move the result. Check 4 passed with about 1.4 percentage points of margin. MLP recomputation is off and unmeasured.” |
| 2:34–2:45 | Speaker 4 | Closing card: “Next: Kaggle T4 run, repeat measurements, measure recomputation separately.” | “Next, complete the Kaggle run when access is available, repeat the measurements, and evaluate recomputation separately. Until then, these remain Colab results.” |

## Commands and output to capture

Show these commands in the terminal or notebook. Capture the results from the prepared GPU run; do not imply the local Mac ran CUDA tests or that the recorded numbers came from Kaggle.

```bash
python -m pytest solution -q
bash scripts/kaggle_3_full.sh
```

Show the recorded solution-test result (37 passed on the T4), then the full measurement summary and check table. The table should show all ten checks passing. Keep the recorded run labeled **Google Colab Tesla T4**. The final measured values are in the README results table; show no invented or rerun values.

## Recording checklist

- [ ] Resolution: TODO — choose and confirm a readable recording resolution.
- [ ] Font size: TODO — choose and check readability at final playback; the exact size is not specified in the README.
- [ ] Hide tokens, API keys, notebook secrets, and personal account details.
- [ ] Show the GPU name, Tesla T4; crop other changing `nvidia-smi` values if they are not part of the story.
- [ ] Label the results as Colab, not Kaggle.
- [ ] Keep the recorded narration and screen capture within 2:45 target and 3:00 maximum.
