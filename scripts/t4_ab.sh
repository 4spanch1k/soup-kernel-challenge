#!/usr/bin/env bash
# Paired solution-only runs of two refs in one Colab session (A-B-A-B), on the public rows and on an
# all-512-token edge file. Run a COPY of this script (it checks out other refs):  cp scripts/t4_ab.sh /tmp/ && bash /tmp/t4_ab.sh
set -u
REPO=${REPO:-/content/repo}
A=${A:-origin/main}
B=${B:-origin/x/causal-attn}
cd "$REPO" || exit 1
python scripts/make_edge512.py /content/edge512.jsonl
summary() {
  python - "$1" <<'PY'
import json, sys
r = json.load(open(sys.argv[1] + "/result.json"))["summary"].get("solution", {})
print(f"   => wall {r.get('wall_s')} s  peak {r.get('gpu_peak_mib')} MiB  eval loss {r.get('eval_loss')}  ok={r.get('ok')}")
PY
}
run() {  # tag ref [train-file]
  git checkout -q "$2" || return
  echo "### $1  ($2, $(git rev-parse --short HEAD))"
  if [ -n "${3:-}" ]; then
    python harness/run_task.py --only solution --out "runs_ab/$1" --train-file "$3" > "/tmp/ab_$1.log" 2>&1
  else
    python harness/run_task.py --only solution --out "runs_ab/$1" > "/tmp/ab_$1.log" 2>&1
  fi
  summary "runs_ab/$1"
}
run edge_A "$A" /content/edge512.jsonl
run edge_B "$B" /content/edge512.jsonl
run pub_A1 "$A"
run pub_B1 "$B"
run pub_A2 "$A"
run pub_B2 "$B"
git checkout -q "$B"
echo "### done"
