#!/usr/bin/env bash
# Redirect: run the c17 memory-peak diagnostic (public data) from this notebook.
set -u
cd "$(dirname "$0")/.."
git fetch -q origin c17/mempeak
git checkout -q -f FETCH_HEAD
bash scripts/t4_mem2.sh public 2>&1 | grep -a "^\[mem\]" | cut -c1-170 | head -110
echo "=== done ==="
