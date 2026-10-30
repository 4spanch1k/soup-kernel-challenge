"""How much of each padded batch is real tokens? Tokenizes the public training rows like Soup does
(chat template, 512 max) and simulates random batches of 4 padded to their longest row. Also reads
the real cumulative `num_tokens` of the last solution run when it exists."""
import glob
import json
import random

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
lengths = []
for line in open("data/train.jsonl", encoding="utf-8"):
    row = json.loads(line)
    extra = row.get("input") or ""
    user = f"{row['instruction']}\n{extra}".strip() if extra else row["instruction"]
    ids = tok.apply_chat_template([{"role": "user", "content": user},
                                   {"role": "assistant", "content": row["output"]}],
                                  tokenize=True, return_dict=False)
    lengths.append(min(len(ids), 512))
print("rows", len(lengths), "mean len", sum(lengths) / len(lengths), "max", max(lengths))
random.seed(0)
valid = padded = 0
for _ in range(3000):
    batch = random.sample(lengths, 4)
    valid += sum(batch)
    padded += 4 * max(batch)
print(f"simulated valid/padded = {valid / padded:.3f}  (padding share {1 - valid / padded:.1%})")
for path in sorted(glob.glob("runs_try/solution_1/adapter/checkpoint-*/trainer_state.json"))[-1:]:
    hist = json.load(open(path))["log_history"]
    nt = [h["num_tokens"] for h in hist if "num_tokens" in h]
    print("real valid tokens over", len(nt), "steps:", nt[-1] if nt else None)
    print("-> valid per step:", nt[-1] / len(nt) if nt else None)
