#!/usr/bin/env bash
# Paired solution-only runs: Triton warm-up thread (SOUP_X_WARM=1) against off, A-B-A-B. One result line per run;
# the [warm] compile log (SOUP_X_WARM_LOG=1) of each run is printed so uncovered kernel variants are visible.
set -u
cd "$(dirname "$0")/.." || exit 1
run_case() {  # tag warm
  local out="runs_exp7/$1"
  echo "=== $1 (SOUP_X_WARM=$2) ==="
  SOUP_X_WARM="$2" SOUP_X_WARM_LOG=1 python -u harness/run_task.py --only solution --out "$out"
  grep -rh "^\[warm\]" "$out" 2>/dev/null | sort -u | head -80
  python -u - "$out/result.json" "$1" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]["solution"]
print(f"[result] {sys.argv[2]} wall_s={s.get('wall_s')} gpu_peak_mib={s.get('gpu_peak_mib')} eval_loss={s.get('eval_loss')}", flush=True)
PY
}
run_case A1 0
run_case W1 1
run_case A2 0
run_case W2 1
echo "=== done ==="
