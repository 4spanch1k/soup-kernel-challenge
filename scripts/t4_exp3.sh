#!/usr/bin/env bash
# Redirect: full measurement and the ten checks for the final candidate branch.
set -u
cd "$(dirname "$0")/.."
git fetch -q origin final/packed-fix
git checkout -q -f FETCH_HEAD
git log -1 --oneline
bash scripts/kaggle_3_full.sh 2>&1 | tail -60
echo "=== done ==="
