#!/usr/bin/env bash
# Redirect: this Kaggle notebook has the GPU session set up, so it runs the c12/warm experiment from here.
# The original paired script is scripts/t4_exp3_orig.sh.
set -eu
cd "$(dirname "$0")/.."
git fetch -q origin c12/warm
git checkout -q -f FETCH_HEAD
exec bash scripts/t4_exp7.sh
