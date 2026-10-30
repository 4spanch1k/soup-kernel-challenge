#!/usr/bin/env bash
# Kaggle T4, step 3: the full measurement (base vs solution), the 10 checks and the score.
set -u
cd "$(dirname "$0")/.."
python harness/run_task.py 2>&1 | tail -30
python harness/check_rules.py 2>&1 | tail -30
python harness/score.py runs/result.json 2>&1 | tail -20
