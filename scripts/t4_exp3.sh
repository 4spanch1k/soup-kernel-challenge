#!/usr/bin/env bash
# Paired A-B-A-B solution-only runs for the three optional tail/startup changes.
set -euo pipefail
cd "$(dirname "$0")/.."

run_case() {
    local tag="$1"
    local switch="$2"

    export SOUP_X_LORA=0
    export SOUP_X_FPRINT=0
    export SOUP_X_MERGE_VARIANTS=0
    case "$switch" in
        off) ;;
        lora) export SOUP_X_LORA=1 ;;
        fprint) export SOUP_X_FPRINT=1 ;;
        merge) export SOUP_X_MERGE_VARIANTS=1 ;;
        all)
            export SOUP_X_LORA=1
            export SOUP_X_FPRINT=1
            export SOUP_X_MERGE_VARIANTS=1
            ;;
        *) printf 'unknown switch: %s\n' "$switch" >&2; return 2 ;;
    esac

    printf '\n=== %s (%s) ===\n' "$tag" "$switch"
    python harness/run_task.py --only solution --out "runs_exp3/$tag"
    python -c 'import json,sys; m=json.load(open(sys.argv[1], encoding="utf-8"))["summary"]["solution"]; print("[result] wall_s={} gpu_peak_mib={} eval_loss={}".format(m.get("wall_s"), m.get("gpu_peak_mib"), m.get("eval_loss")), flush=True)' \
        "runs_exp3/$tag/result.json"
}

for variant in lora fprint merge all; do
    run_case "${variant}_A1" off
    run_case "${variant}_B1" "$variant"
    run_case "${variant}_A2" off
    run_case "${variant}_B2" "$variant"
done
