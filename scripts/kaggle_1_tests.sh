#!/usr/bin/env bash
# Kaggle T4, step 1: install the pinned packages, prepare data, run our kernel tests and Soup's tests.
set -u
cd "$(dirname "$0")/.."
python -m pip install -q "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest 2>&1 | tail -3
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
python task/prepare_data.py 2>&1 | tail -3
echo "=== our kernel tests ==="
python -m pytest solution -q -x -p no:cacheprovider 2>&1 | tail -60
echo "=== soup regression tests with apply() ==="
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q -p no:cacheprovider 2>&1 | tail -15
