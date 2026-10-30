#!/usr/bin/env bash
# Cheap T4 measurements, no code changes: tensor cores in Triton, SDPA backends, profile of our run.
# Run after scripts/kaggle_1_tests.sh (packages + data). Output goes to stdout; copy it back.
set -u
cd "$(dirname "$0")/.."
echo "=== 1. tl.dot on this GPU ===";  python scripts/t4_dot_check.py 2>&1 | tail -20
echo "=== 2. SDPA backends ===";       python scripts/t4_sdpa_check.py 2>&1 | tail -30
echo "=== 3. profile of our run ===";  python harness/run_task.py --only solution --out runs_try --profile > /tmp/profile_run.log 2>&1; tail -5 /tmp/profile_run.log
python scripts/t4_profile_report.py runs_try 2>&1 | tail -60
