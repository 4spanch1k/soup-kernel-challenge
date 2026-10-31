#!/usr/bin/env bash
# Paired solution-only runs: adaptive MLP recompute (SOUP_X_ADAPT_RECOMP=<rows>) against off. Prints one line per run.
set -u
cd "$(dirname "$0")/.." || exit 1
run_case() {  # tag threshold
  local out="runs_exp5/$1"
  echo "=== $1 (SOUP_X_ADAPT_RECOMP=$2) ==="
  SOUP_X_ADAPT_RECOMP="$2" python -u harness/run_task.py --only solution --out "$out"
  python -u - "$out/result.json" "$1" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]["solution"]
print(f"[result] {sys.argv[2]} wall_s={s.get('wall_s')} gpu_peak_mib={s.get('gpu_peak_mib')} eval_loss={s.get('eval_loss')}", flush=True)
PY
}
run_case A1 0
for th in 1500 1100 800; do
  run_case "T${th}_B1" "$th"
  run_case "A_for_T${th}" 0
done
echo "=== done ==="
