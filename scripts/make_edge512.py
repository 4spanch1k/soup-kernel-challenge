"""Edge-case training file built ONLY from the public rows (data/train.jsonl): every row is stretched by
repeating its own answer until it is longer than the 512-token limit, so every batch has no padding.
    python scripts/make_edge512.py [out.jsonl]"""
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/edge512.jsonl")
rows = [json.loads(line) for line in open(repo / "data" / "train.jsonl", encoding="utf-8")]
with open(out, "w", encoding="utf-8") as handle:
    for row in rows:
        text = row["output"] or "x"
        while len(text) < 6000:
            text += " " + row["output"]
        row = {"instruction": row["instruction"], "input": row["input"], "output": text}
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} rows to {out}")
