"""Summarise the last profiled solution run (scripts/kaggle_2_quick.sh): versions, memory, loop overhead,
top GPU kernels and what the attention kernels are."""
import glob
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

base = sys.argv[1] if len(sys.argv) > 1 else "runs_try"
try:
    metrics = json.load(open(f"{base}/solution_1/metrics.json"))
    print("METRICS", {key: metrics.get(key) for key in (
        "train_loop_s", "torch_max_allocated_mib", "torch_max_reserved_mib", "cuda_device", "alloc_conf")})
    own = metrics.get("own_kernel_ms")
    print("own_kernel_ms:", own)
    steps = metrics.get("step_s") or metrics.get("steps")
    if steps and metrics.get("train_loop_s"):
        values = [s if isinstance(s, (int, float)) else s.get("step_s", 0) for s in steps]
        print(f"train_loop_s - sum(step_s) = {metrics['train_loop_s'] - sum(values):.2f} s")
except Exception as exc:  # noqa: BLE001
    print("metrics unavailable:", exc)

path = f"{base}/solution_profile/profile.json"
data = json.load(open(path))
n, total = data["active_steps"], data["total_gpu_us"]
print(f"\ntotal GPU ms/step: {total / n / 1000:.1f}")
attention = 0.0
for row in data["kernels"][:40]:
    ms = row["time_us"] / n / 1000
    name = row["name"]
    tag = ""
    if any(token in name.lower() for token in ("fmha", "flash", "efficient", "attention", "softmax")):
        tag = "  <-- attention?"
        attention += ms
    print(f'{row["time_us"] / total * 100:5.1f}%  {ms:7.2f} ms/step  n={row["count"] // n:4d}  {name[:95]}{tag}')
print(f"\nattention-looking kernels in the top 40: {attention:.1f} ms/step")
