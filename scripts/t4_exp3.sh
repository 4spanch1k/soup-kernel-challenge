#!/usr/bin/env bash
# Collector: look for the varlen messages in the logs that earlier runs left on this VM.
set -u
for d in /kaggle/working/soup-exp3.*/repo/runs_exp9/V1 /kaggle/working/soup-exp3.*/repo/runs_exp9/V2; do
  [ -d "$d" ] || continue
  echo "=== $d ==="
  find "$d" -name "*.log" | head -5
  grep -rao "\[varlen\][^\r]\{0,200\}" "$d" 2>/dev/null | sort | uniq -c | head -10
  echo "-- count of 'varlen' mentions:"; grep -rac "varlen" "$d" 2>/dev/null | head -5
done
echo "=== collected ==="
