#!/usr/bin/env python3
"""Summarize the non-step startup and shutdown time from one t4_diag2 run."""

import argparse
import json
import re
import statistics
from pathlib import Path


IMPORT_LINE = re.compile(r"^import time:\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.*)$")
IMPORTS = ("torch", "transformers", "trl", "peft", "bitsandbytes", "triton", "datasets")


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _import_times(log_path):
    found = {}
    if not log_path.exists():
        return found
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = IMPORT_LINE.match(line)
        if not match:
            continue
        name = match.group(3).strip()
        if name in IMPORTS:
            found[name] = {
                "self_s": int(match.group(1)) / 1_000_000,
                "cumulative_s": int(match.group(2)) / 1_000_000,
            }
    return found


def analyze(run_dir):
    result = _read_json(run_dir / "result.json") or {}
    solutions = result.get("runs", {}).get("solution", [])
    run_name = solutions[0].get("run") if solutions else "solution_1"
    run_dir = run_dir / run_name
    metrics = _read_json(run_dir / "metrics.json") or {}
    probe = _read_json(run_dir.parent / "startup_probe.json") or {}
    run_wall_s = solutions[0].get("wall_s") if solutions else None
    train_loop_s = metrics.get("train_loop_s")
    step_s = metrics.get("step_s", [])
    apply_s = metrics.get("apply_s")

    first = probe.get("first_step_begin")
    last = probe.get("last_step_end")
    process_start = probe.get("process_start")
    process_end = probe.get("process_end")
    process_s = process_end - process_start if process_end and process_start else None
    before_loop_s = first - process_start if first and process_start else None
    after_loop_s = process_end - last if process_end and last else None
    outside_loop_s = run_wall_s - train_loop_s if run_wall_s is not None and train_loop_s else None
    after_apply_s = max(0.0, before_loop_s - apply_s) if before_loop_s is not None and apply_s else None
    stable = step_s[10:] if len(step_s) > 10 else step_s[3:]
    first_three_excess_s = None
    if len(step_s) >= 4 and stable:
        first_three_excess_s = max(0.0, sum(step_s[:3]) - 3 * statistics.median(stable))

    import_times = _import_times(run_dir / "train.log")
    report = {
        "status": "measured from one harness run",
        "run_wall_s": run_wall_s,
        "process_wall_s": round(process_s, 3) if process_s is not None else None,
        "before_first_step_s": round(before_loop_s, 3) if before_loop_s is not None else None,
        "solution_import_and_apply_s": apply_s,
        "after_apply_before_first_step_s": round(after_apply_s, 3)
        if after_apply_s is not None else None,
        "train_loop_s": train_loop_s,
        "after_last_step_s": round(after_loop_s, 3) if after_loop_s is not None else None,
        "wall_minus_train_loop_s": round(outside_loop_s, 3)
        if outside_loop_s is not None else None,
        "first_three_step_excess_estimate_s": round(first_three_excess_s, 3)
        if first_three_excess_s is not None else None,
        "first_three_step_estimate_note":
            "ОЦЕНКА: includes batch-shape drift and any first-step compilation; not compile-only",
        "importtime": import_times,
        "importtime_note": "Package cumulative values include nested imports and must not be added.",
        "probe_errors": probe.get("errors", []),
    }
    (run_dir.parent / "startup.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[startup] run_wall_s={report['run_wall_s']}", flush=True)
    print(f"[startup] before_first_step_s={report['before_first_step_s']}", flush=True)
    print(f"[startup] solution_import_and_apply_s={report['solution_import_and_apply_s']}",
          flush=True)
    print(f"[startup] after_apply_before_first_step_s={report['after_apply_before_first_step_s']}",
          flush=True)
    print(f"[startup] train_loop_s={report['train_loop_s']}", flush=True)
    print(f"[startup] after_last_step_s={report['after_last_step_s']}", flush=True)
    print(f"[startup] wall_minus_train_loop_s={report['wall_minus_train_loop_s']}", flush=True)
    print(f"[startup] first_three_step_excess_estimate_s="
          f"{report['first_three_step_excess_estimate_s']} (ОЦЕНКА)", flush=True)
    if import_times:
        for package, times in import_times.items():
            print(f"[startup] import={package} self_s={times['self_s']:.3f} "
                  f"cumulative_s={times['cumulative_s']:.3f} (nested imports included)", flush=True)
    else:
        print("[startup] importtime entries not found in train.log", flush=True)
    print(f"[startup] json={run_dir.parent / 'startup.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="output directory used by scripts/t4_diag2.sh")
    args = parser.parse_args()
    analyze(args.run_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
