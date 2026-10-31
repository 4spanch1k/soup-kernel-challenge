#!/usr/bin/env bash
# One run with the single-call attention path (SOUP_X_VARLEN=1) and the [varlen] log, then one run without. Compact output.
set -u
cd "$(dirname "$0")/.." || exit 1
run_case() {  # tag varlen
  local out="runs_exp9/$1"
  echo "=== $1 (SOUP_X_VARLEN=$2) ==="
  SOUP_X_VARLEN="$2" SOUP_X_MERGE_VARIANTS=0 SOUP_X_WARM=0 SOUP_X_WARM_LOG=1 python -u harness/run_task.py --only solution --out "$out" > "$out.log" 2>&1 || true
  grep -rh "^\[varlen\]" "$out" 2>/dev/null | sort -u | head -8
  python3 - "$out/result.json" "$1" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))["summary"]["solution"]
    print(f"[result] {sys.argv[2]} wall_s={s.get('wall_s')} gpu_peak_mib={s.get('gpu_peak_mib')} eval_loss={s.get('eval_loss')}", flush=True)
except Exception as exc:
    print(f"[result] {sys.argv[2]} unavailable: {exc!r}", flush=True)
PY
}
mkdir -p runs_exp9
run_case V1 1
run_case A1 0
run_case V2 1
echo "=== done ==="
