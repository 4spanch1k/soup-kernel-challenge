#!/usr/bin/env bash
# Run after scripts/kaggle_1_tests.sh on a T4. Repository outputs go under runs_exp/.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p runs_exp
public_file="$repo_root/data/train.jsonl"
if [[ ! -f "$public_file" ]]; then
  printf 'missing public data: %s (run scripts/kaggle_1_tests.sh first)\n' "$public_file" >&2
  exit 1
fi

printf '%s\n' '=== solution CUDA tests (all experiment switches off) ==='
env -u SOUP_X_ATTN_CKPT -u SOUP_X_EXACT_RESID -u SOUP_X_MERGE_VARIANTS \
  python -B -u -m pytest solution -q -p no:cacheprovider

edge_file="$repo_root/runs_exp/edge512.jsonl"
python -B -u scripts/make_edge512.py "$edge_file"

run_case() {
  local dataset="$1"
  local variant="$2"
  local arm="$3"
  local repeat="$4"
  local train_file="$5"
  local out="runs_exp/${dataset}_${variant}_${arm}${repeat}"
  local -a experiment_env=()

  if [[ "$arm" == "B" ]]; then
    case "$variant" in
      attn1) experiment_env=(SOUP_X_ATTN_CKPT=1) ;;
      attn2) experiment_env=(SOUP_X_ATTN_CKPT=2) ;;
      resid) experiment_env=(SOUP_X_EXACT_RESID=1) ;;
      variants) experiment_env=(SOUP_X_MERGE_VARIANTS=1) ;;
      all) experiment_env=(SOUP_X_ATTN_CKPT=1 SOUP_X_EXACT_RESID=1 SOUP_X_MERGE_VARIANTS=1) ;;
      *) printf 'unknown variant: %s\n' "$variant" >&2; return 2 ;;
    esac
  fi

  printf '[t4_exp] %s %s %s%s\n' "$dataset" "$variant" "$arm" "$repeat"
  if [[ "$arm" == "A" ]]; then
    env -u SOUP_X_ATTN_CKPT -u SOUP_X_EXACT_RESID -u SOUP_X_MERGE_VARIANTS \
      python -B -u harness/run_task.py --only solution --out "$out" --train-file "$train_file"
  else
    env -u SOUP_X_ATTN_CKPT -u SOUP_X_EXACT_RESID -u SOUP_X_MERGE_VARIANTS \
      "${experiment_env[@]}" python -B -u harness/run_task.py --only solution --out "$out" \
      --train-file "$train_file"
  fi

  python -B -u - "$out/result.json" "$dataset/$variant/$arm$repeat" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as result_file:
    result = json.load(result_file)
summary = result["summary"]["solution"]
print(f"[result] {sys.argv[2]} wall_s={summary['wall_s']} "
      f"gpu_peak_mib={summary['gpu_peak_mib']} eval_loss={summary['eval_loss']}",
      flush=True)
PY
}

printf '%s\n' '=== T4 diagnostics (first) ==='
REPO="$repo_root" bash scripts/t4_diag.sh || true

# trimmed run list (about 40 minutes): the full package first, then each switch alone
for variant in all attn1 resid variants attn2; do
  run_case public "$variant" A 1 "$public_file"
  run_case public "$variant" B 1 "$public_file"
  run_case public "$variant" A 2 "$public_file"
  run_case public "$variant" B 2 "$public_file"
done
# dense batches: one A-B pair of the full package
run_case edge512 all A 1 "$edge_file"
run_case edge512 all B 1 "$edge_file"

printf '%s\n' '=== done ==='
