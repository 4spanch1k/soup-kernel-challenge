#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '[idle] requires a Kaggle CUDA runtime with nvidia-smi\n' >&2
    exit 2
fi

python_bin="${PYTHON:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    python_bin="python3"
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf '[idle] python or python3 was not found\n' >&2
    exit 2
fi

if [[ -n "${T4_DIAG_OUT:-}" ]]; then
    "$python_bin" -X importtime scripts/t4_idle.py --out "$T4_DIAG_OUT"
else
    "$python_bin" -X importtime scripts/t4_idle.py
fi
