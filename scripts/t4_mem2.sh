#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-dense}"
if [[ "$mode" != "public" && "$mode" != "dense" ]]; then
    echo "usage: bash scripts/t4_mem2.sh [public|dense]" >&2
    exit 2
fi

if [[ ! -f data/train.jsonl || ! -f data/eval_public.jsonl ]]; then
    python task/prepare_data.py
fi

output_dir="${T4_MEM_OUT:-/kaggle/working/t4_mempeak_${mode}}"
train_file="$PWD/data/train.jsonl"

if [[ "$mode" == "dense" ]]; then
    train_file="$output_dir/train_4x512.jsonl"
    mkdir -p "$output_dir"
    python - "$PWD/data/train.jsonl" "$train_file" <<'PY'
import json
import sys

source, target = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle]

with open(target, "w", encoding="utf-8") as handle:
    for row in rows:
        answer = row["output"] or "x"
        text = answer
        while len(text) < 6000:
            text += " " + answer
        dense_row = {
            "instruction": row["instruction"],
            "input": row["input"],
            "output": text,
        }
        handle.write(json.dumps(dense_row, ensure_ascii=False) + "\n")

print(f"[mem] dense_rows={len(rows)} train_file={target}", flush=True)
PY
fi

python scripts/t4_mempeak.py \
    --out "$output_dir/runs" \
    --train-file "$train_file" \
    --eval-file "$PWD/data/eval_public.jsonl"

train_log="$output_dir/runs/solution_1/train.log"
if [[ -f "$train_log" ]]; then
    grep -E '^\[mem\]' "$train_log" || true
fi
