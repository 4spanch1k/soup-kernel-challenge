#!/usr/bin/env bash
# Collector: print what earlier runs of this Kaggle VM left in /kaggle/working (runs_exp7 / runs_exp8 result files and logs).
set -u
for d in /kaggle/working/soup-exp3.*/repo/runs_exp8/* /kaggle/working/soup-exp3.*/repo/runs_exp7/*; do
  [ -d "$d" ] || continue
  echo "=== $d ==="
  if [ -f "$d/result.json" ]; then
    python3 - "$d/result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
s = r.get("summary", {}).get("solution", {})
print("[collect]", {k: s.get(k) for k in ("wall_s", "gpu_peak_mib", "eval_loss")}, flush=True)
PY
  else
    echo "[collect] no result.json yet"
  fi
  grep -rh "^\[varlen\]\|^\[warm\]" "$d" 2>/dev/null | sort -u | head -12
done
ls -d /kaggle/working/soup-exp3.* 2>/dev/null
tail -n 5 /kaggle/working/exp3.log 2>/dev/null
echo "=== collected ==="
