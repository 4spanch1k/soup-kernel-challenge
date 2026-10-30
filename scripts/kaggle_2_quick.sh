#!/usr/bin/env bash
# Kaggle T4, step 2: quick run of the solution alone, with a per-kernel GPU profile.
set -u
cd "$(dirname "$0")/.."
python harness/run_task.py --only solution --out runs_try --profile 2>&1 | tail -40
python - <<'PY'
import json
m = json.load(open("runs_try/solution_1/metrics.json")) if __import__("os").path.exists("runs_try/solution_1/metrics.json") else {}
print("own_kernel_ms:", m.get("own_kernel_ms"))
print("timed_loop_s:", m.get("timed_loop_s"), "peak_reserved_mib:", m.get("torch_max_reserved_mib"))
PY
