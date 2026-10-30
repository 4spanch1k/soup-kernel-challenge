"""Print library versions and the top GPU kernels of the last profiled solution run
(runs_try/solution_profile/profile.json, written by scripts/kaggle_2_quick.sh)."""
import json
import sys

import torch

versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
for name in ("triton", "transformers", "peft", "trl", "bitsandbytes", "accelerate"):
    try:
        versions[name] = __import__(name).__version__
    except Exception as exc:  # noqa: BLE001
        versions[name] = f"? {exc.__class__.__name__}"
print("VERSIONS", versions)
path = sys.argv[1] if len(sys.argv) > 1 else "runs_try/solution_profile/profile.json"
data = json.load(open(path))
steps = data["active_steps"]
total = data["total_gpu_us"]
print(f"total GPU ms/step: {total / steps / 1000:.1f}")
for k in data["kernels"][:32]:
    print(f'{k["time_us"] / total * 100:5.1f}%  {k["time_us"] / steps / 1000:7.2f} ms/step  n={k["count"] // steps:4d}  {k["name"][:100]}')
