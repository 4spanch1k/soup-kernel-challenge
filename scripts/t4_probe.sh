#!/usr/bin/env bash
# Memory probe only (no code changes): per-step valid tokens, memory after the model forward, peak allocated/reserved.
# Run after scripts/kaggle_1_tests.sh. Prints the table (or the error tail) for the public rows and the all-512 file.
set -u
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
python scripts/make_edge512.py "$REPO/runs_exp/edge512.jsonl" 2>/dev/null || { mkdir -p runs_exp; python scripts/make_edge512.py "$REPO/runs_exp/edge512.jsonl"; }
probe() {  # tag train-file
  d="/kaggle/working/diag_$1"; rm -rf "$d"; mkdir -p "$d"
  python - "$d" "$2" <<'PY'
import sys
sys.path.insert(0, "harness")
import run_task
from pathlib import Path
run_task.make_config(Path(sys.argv[1]), sys.argv[2], 1)
PY
  echo "### probe $1 ($2)"
  SOUP_DB_PATH=$d/experiments.db TRITON_CACHE_DIR=$d/triton PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    python -u scripts/t4_mem_probe.py --config "$d/task.yaml" --metrics "$d/metrics.json" --solution > "$d/log.txt" 2>&1
  if grep -q "^MEMPROBE" "$d/log.txt"; then grep -A14 "^MEMPROBE" "$d/log.txt"; else echo "NO TABLE, tail of log:"; tail -25 "$d/log.txt"; fi
}
probe pub "$REPO/data/train.jsonl"
probe edge "$REPO/runs_exp/edge512.jsonl"
echo "=== probe done ==="
