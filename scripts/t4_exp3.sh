#!/usr/bin/env bash
# Redirect: run the c14/combo varlen check from this notebook (its GPU session is set up).
set -eu
cd "$(dirname "$0")/.."
git fetch -q origin c14/combo
git checkout -q -f FETCH_HEAD
exec bash scripts/t4_exp9.sh
