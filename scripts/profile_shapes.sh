#!/usr/bin/env bash
# Debug only: profile the solution run with tensor shapes, to see which cast/copy/add ops are the
# expensive ones. Uses a patched COPY of harness/_train.py (the harness itself is not touched).
set -u
cd "$(dirname "$0")/.."
sed -e 's/schedule=schedule(wait=PROFILE_WAIT, warmup=PROFILE_WARMUP,/record_shapes=True, schedule=schedule(wait=PROFILE_WAIT, warmup=PROFILE_WARMUP,/' \
    harness/_train.py > harness/_train_dbg.py
cat >> harness/_train_dbg.py <<'PY'


def _shape_report(prof):
    rows = []
    for e in prof.key_averages(group_by_input_shape=True):
        if e.key in ("aten::copy_", "aten::to", "aten::_to_copy", "aten::add", "aten::add_", "aten::mul",
                     "aten::contiguous", "aten::clone", "aten::dropout", "aten::native_dropout"):
            t = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
            rows.append((t, e.count, e.key, str(e.input_shapes)[:110]))
    rows.sort(reverse=True)
    print("TOP ops by GPU time, per step (16 steps):")
    for t, c, k, s in rows[:22]:
        print(f"{t/16/1000:7.2f} ms/step n={c//16:4d} {k:18s} {s}")
PY
sed -i 's/        table = kernel_table(prof)/        _shape_report(prof)\n        table = kernel_table(prof)/' harness/_train_dbg.py
CFG=runs_try/solution_profile/task.yaml
[ -f "$CFG" ] || { echo "run scripts/kaggle_2_quick.sh first (needs $CFG)"; exit 1; }
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SOUP_NO_AUDIT_LOG=1 \
  python harness/_train_dbg.py --config "$CFG" --metrics /tmp/dbg_metrics.json --solution --profile /tmp/dbg_profile.json > /tmp/dbg.log 2>&1
grep -A30 "TOP ops" /tmp/dbg.log || { echo "NO REPORT; log tail:"; tail -25 /tmp/dbg.log; }
