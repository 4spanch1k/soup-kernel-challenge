#!/usr/bin/env bash
# T4 diagnostics (no code changes): memory probe on the public rows and on the all-512 edge file, then the head dX GEMM bench.
# Run a copy:  cp scripts/t4_diag.sh /tmp/ && bash /tmp/t4_diag.sh
set -u
cd "${REPO:-/content/repo}" || exit 1
python scripts/make_edge512.py /content/edge512.jsonl >/dev/null 2>&1 || true
probe() {  # tag train-file
  d=/tmp/diag_$1; rm -rf "$d"; mkdir -p "$d"
  python - "$d" "$2" <<'PY'
import sys, os
sys.path.insert(0, "harness")
import run_task
from pathlib import Path
run_dir = Path(sys.argv[1]); cfg = run_task.make_config(run_dir, sys.argv[2], 1)
print(cfg)
PY
  echo "### probe $1 ($2)"
  SOUP_DB_PATH=$d/experiments.db TRITON_CACHE_DIR=$d/triton PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    python scripts/t4_mem_probe.py --config "$d/task.yaml" --metrics "$d/metrics.json" --solution > "$d/log.txt" 2>&1
  grep -A14 "^MEMPROBE" "$d/log.txt" | head -16
}
probe pub "$PWD/data/train.jsonl"
probe edge /content/edge512.jsonl
echo "### head dX GEMM bench"
python scripts/t4_gemm_bench.py --sections head --Ms 1170 2>&1 | tail -25
