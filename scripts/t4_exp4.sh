#!/usr/bin/env bash
# Compare the default padded attention path with one varlen attention call on a T4.
set -euo pipefail
cd "$(dirname "$0")/.."

run_one() {
    local tag="$1"
    local enabled="$2"
    local out="runs_exp4/${tag}"
    local status=0

    SOUP_X_VARLEN="$enabled" python harness/run_task.py --only solution --out "$out" \
        --train-file data/train.jsonl || status=$?

    if [[ -f "$out/result.json" ]]; then
        python - "$out/result.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as result_file:
    summary = json.load(result_file)["summary"]["solution"]

print(
    f"[result] wall_s={summary.get('wall_s')} "
    f"gpu_peak_mib={summary.get('gpu_peak_mib')} eval_loss={summary.get('eval_loss')}",
    flush=True,
)
PY
    else
        echo "[result] wall_s=unavailable gpu_peak_mib=unavailable eval_loss=unavailable" >&2
    fi

    return "$status"
}

run_one A1 0
run_one B1 1
run_one A2 0
run_one B2 1
