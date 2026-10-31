#!/usr/bin/env bash
# Paired solution-only runs: all optional experiments off against the requested combo, A-B-A-B.
set -u
cd "$(dirname "$0")/.." || exit 1

failed=0
run_case() {
    local tag="$1"
    local variant="$2"
    local out="runs_exp8/$tag"
    local warm=0
    local fprint=0
    local merge_variants=0
    local varlen=0
    local status=0

    local adapt=0
    if [[ "$variant" == ADAPT ]]; then
        adapt=1100
    fi
    if [[ "$variant" == COMBO ]]; then
        warm=0  # warm-up thread measured on T4: no gain beyond noise, left out
        fprint=0  # fixed Dataset.map fingerprint: judged doubtful by review, kept out of the combo
        merge_variants=1
        varlen=1
    fi

    echo "=== $tag ($variant) ==="
    SOUP_X_WARM="$warm" \
    SOUP_X_WARM_LOG="$warm" \
    SOUP_X_FPRINT="$fprint" \
    SOUP_X_MERGE_VARIANTS="$merge_variants" \
    SOUP_X_VARLEN="$varlen" \
    SOUP_X_ADAPT_RECOMP="$adapt" \
    SOUP_X_LORA=0 \
    SOUP_RECOMPUTE_MLP=0 \
        python -u harness/run_task.py --only solution --out "$out" || status=$?

    grep -rh "^\[warm\]\|^\[varlen\]" "$out" 2>/dev/null | sort -u | head -60
    if [[ -f "$out/result.json" ]]; then
        python -u - "$out/result.json" "$tag" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as result_file:
    summary = json.load(result_file)["summary"]["solution"]

print(
    f"[result] {sys.argv[2]} wall_s={summary.get('wall_s')} "
    f"gpu_peak_mib={summary.get('gpu_peak_mib')} eval_loss={summary.get('eval_loss')}",
    flush=True,
)
PY
    else
        echo "[result] $tag wall_s=unavailable gpu_peak_mib=unavailable eval_loss=unavailable"
    fi

    return "$status"
}

run_case A1 A || failed=1
run_case COMBO1 COMBO || failed=1
run_case A2 A || failed=1
run_case COMBO2 COMBO || failed=1
run_case ADAPT1 ADAPT || failed=1
run_case A3 A || failed=1
echo "=== done ==="
exit "$failed"
