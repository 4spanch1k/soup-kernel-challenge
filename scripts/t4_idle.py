#!/usr/bin/env python3
"""Run one solution-only Soup training pass and profile steps 10-14 on a Kaggle T4."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def install_site_probe():
    """Called by the temporary sitecustomize module in harness/_train.py only."""
    if getattr(sys, "_soup_idle_probe", False):
        return
    sys._soup_idle_probe = True

    import atexit
    import builtins

    torch = None
    state = {
        "process_start": time.perf_counter(),
        "first_step_begin": None,
        "last_step_end": None,
        "profiled_steps": {},
        "errors": [],
    }
    profile_path = Path(os.environ["SOUP_DIAG_TRACE_PATH"])
    probe_path = Path(os.environ["SOUP_DIAG_PROBE_PATH"])
    active_profile = None
    marker = None
    active_step = None
    step_started = None

    def save_probe():
        state["process_end"] = time.perf_counter()
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        probe_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    atexit.register(save_probe)

    def add_callback():
        nonlocal torch
        import transformers

        torch = __import__("torch")

        class IdleProbe(transformers.TrainerCallback):
            def on_step_begin(self, args, trainer_state, control, **kwargs):
                nonlocal active_profile, marker, active_step, step_started
                step = int(trainer_state.global_step) + 1
                if state["first_step_begin"] is None:
                    state["first_step_begin"] = time.perf_counter()
                active_step = step
                if step == 10:
                    try:
                        from torch.profiler import ProfilerActivity, profile

                        active_profile = profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            record_shapes=False,
                            profile_memory=False,
                            with_stack=False,
                        )
                        active_profile.start()
                    except Exception as exc:  # profiling failure must not stop the training run
                        state["errors"].append(f"profiler start: {exc!r}")
                        active_profile = None
                if 10 <= step <= 14:
                    if active_profile is not None:
                        try:
                            marker = torch.profiler.record_function(f"idle_step_{step}")
                            marker.__enter__()
                        except Exception as exc:
                            state["errors"].append(f"step {step} marker: {exc!r}")
                            marker = None
                    step_started = time.perf_counter()

            def on_step_end(self, args, trainer_state, control, **kwargs):
                nonlocal active_profile, marker, active_step, step_started
                final_step = int(getattr(args, "max_steps", trainer_state.max_steps))
                final = trainer_state.global_step >= final_step or control.should_training_stop
                profiled = active_step is not None and 10 <= active_step <= 14
                if profiled or final:
                    ended = time.perf_counter()
                if final:
                    state["last_step_end"] = ended
                if profiled and step_started is not None:
                    state["profiled_steps"][str(active_step)] = round(
                        (ended - step_started) * 1000, 3
                    )
                if marker is not None:
                    try:
                        marker.__exit__(None, None, None)
                    except Exception as exc:
                        state["errors"].append(f"step {active_step} marker close: {exc!r}")
                    marker = None
                if active_step == 14 and active_profile is not None:
                    try:
                        active_profile.stop()
                        profile_path.parent.mkdir(parents=True, exist_ok=True)
                        active_profile.export_chrome_trace(str(profile_path))
                    except Exception as exc:
                        state["errors"].append(f"profiler export: {exc!r}")
                    active_profile = None
                active_step = None
                step_started = None

        original_init = transformers.Trainer.__init__

        def init(trainer, *args, **kwargs):
            original_init(trainer, *args, **kwargs)
            trainer.add_callback(IdleProbe())

        transformers.Trainer.__init__ = init

    original_import = __import__

    def import_with_probe(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        if name == "soup_cli.cli":
            try:
                add_callback()
            except Exception as exc:  # instrumentation failure must not change training behavior
                state["errors"].append(f"callback setup: {exc!r}")
            finally:
                builtins.__import__ = original_import
        return module

    builtins.__import__ = import_with_probe


def _kernel_events(trace):
    return [
        event for event in trace.get("traceEvents", [])
        if event.get("ph") == "X"
        and "kernel" in str(event.get("cat", "")).lower()
        and "memcpy" not in str(event.get("name", "")).lower()
    ]


def _launch_api_events(trace):
    needles = ("cudalaunchkernel", "culaunchkernel")
    return [
        event for event in trace.get("traceEvents", [])
        if event.get("ph") == "X"
        and any(needle in str(event.get("name", "")).lower() for needle in needles)
    ]


def _union_us(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def summarize_trace(trace_path, probe_path):
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    events = trace.get("traceEvents", [])
    steps = {
        event["name"]: event
        for event in events
        if event.get("ph") == "X" and str(event.get("name", "")).startswith("idle_step_")
    }
    kernels = _kernel_events(trace)
    launch_apis = _launch_api_events(trace)
    step_rows = []

    for step_num in range(10, 15):
        name = f"idle_step_{step_num}"
        step = steps.get(name)
        if step is None:
            continue
        start, end = float(step["ts"]), float(step["ts"]) + float(step.get("dur", 0))
        contained_kernels = [
            event for event in kernels
            if float(event["ts"]) < end
            and float(event["ts"]) + float(event.get("dur", 0)) > start
        ]
        contained_launches = [
            event for event in launch_apis if start <= float(event.get("ts", -1)) <= end
        ]
        intervals = [
            (max(start, float(event["ts"])),
             min(end, float(event["ts"]) + float(event.get("dur", 0))))
            for event in contained_kernels
        ]
        active_us = _union_us(intervals)
        span_us = max(end - start, 0.0)
        kernel_sum_us = sum(float(event.get("dur", 0)) for event in contained_kernels)
        launch_cpu_us = sum(float(event.get("dur", 0)) for event in contained_launches)
        step_rows.append({
            "step": step_num,
            "wall_ms": probe.get("profiled_steps", {}).get(str(step_num)),
            "profiler_span_ms": round(span_us / 1000, 3),
            "gpu_kernel_sum_ms": round(kernel_sum_us / 1000, 3),
            "gpu_active_union_ms": round(active_us / 1000, 3),
            "gpu_idle_ms": round(max(span_us - active_us, 0.0) / 1000, 3),
            "gpu_idle_fraction": round(max(span_us - active_us, 0.0) / span_us, 4)
            if span_us else None,
            "cuda_launches": len(contained_kernels),
            "cpu_launch_api_count": len(contained_launches),
            "cpu_launch_api_ms": round(launch_cpu_us / 1000, 3) if contained_launches else None,
            "cpu_ms_per_launch_api": round(launch_cpu_us / len(contained_launches) / 1000, 4)
            if contained_launches else None,
        })

    totals = {}
    for row in step_rows:
        step = steps[f"idle_step_{row['step']}"]
        start, end = float(step["ts"]), float(step["ts"]) + float(step.get("dur", 0))
        for event in kernels:
            if float(event["ts"]) < end and float(event["ts"]) + float(event.get("dur", 0)) > start:
                kernel_name = str(event.get("name", "unknown"))
                item = totals.setdefault(kernel_name, {"name": kernel_name, "time_us": 0.0, "count": 0})
                item["time_us"] += float(event.get("dur", 0))
                item["count"] += 1
    top_kernels = sorted(totals.values(), key=lambda item: item["time_us"], reverse=True)[:15]
    for item in top_kernels:
        item["time_ms"] = round(item.pop("time_us") / 1000, 3)

    return {
        "profiled_steps": step_rows,
        "top_kernels": top_kernels,
        "profile_error": probe.get("errors", []),
        "profiler": "torch.profiler CPU+CUDA, steps 10-14",
        "launch_count_source": "CUDA kernel trace events",
        "cpu_launch_time_source": "cudaLaunchKernel/cuLaunchKernel CPU trace events",
    }


def _terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.poll() is None:
                raise
        process.wait()


def _default_out(repo):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else repo
    return root / f"t4_diag2_{stamp}"


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="new output directory; default is timestamped")
    parser.add_argument("--timeout", type=int, default=660,
                        help="overall run_task limit in seconds (default: 660)")
    args = parser.parse_args()

    if shutil.which("nvidia-smi") is None:
        sys.exit("[idle] requires a Kaggle CUDA runtime with nvidia-smi; no training was started")
    out = (args.out or _default_out(repo)).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"[idle] output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "idle_trace.json"
    probe_path = out / "startup_probe.json"
    print(f"[idle] output_dir={out}", flush=True)
    print("[idle] running exactly one harness solution pass; profiler active on steps 10-14",
          flush=True)

    with tempfile.TemporaryDirectory(prefix="soup_idle_site_") as temp:
        temp_path = Path(temp)
        sitecustomize = temp_path / "sitecustomize.py"
        sitecustomize.write_text(
            "import os, sys\n"
            "if os.environ.get('SOUP_DIAG_IDLE') == '1' and sys.argv and "
            "sys.argv[0].endswith('_train.py'):\n"
            "    from t4_idle import install_site_probe\n"
            "    install_site_probe()\n",
            encoding="utf-8",
        )
        env = dict(os.environ)
        script_path = str(Path(__file__).resolve().parent)
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(temp_path), script_path, env.get("PYTHONPATH", "")) if part
        )
        env["PYTHONPROFILEIMPORTTIME"] = "1"
        env["SOUP_DIAG_IDLE"] = "1"
        env["SOUP_DIAG_TRACE_PATH"] = str(trace_path)
        env["SOUP_DIAG_PROBE_PATH"] = str(probe_path)
        command = [sys.executable, str(repo / "harness" / "run_task.py"),
                   "--only", "solution", "--out", str(out), "--timeout", "600"]
        process = subprocess.Popen(command, cwd=repo, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            _terminate_group(process)
            sys.exit(f"[idle] run_task exceeded {args.timeout}s; process group was stopped")

    if code != 0:
        sys.exit(f"[idle] run_task exited {code}; see {out}")
    profile_complete = False
    if trace_path.exists() and probe_path.exists():
        idle = summarize_trace(trace_path, probe_path)
        (out / "idle.json").write_text(json.dumps(idle, indent=2), encoding="utf-8")
        profile_complete = len(idle["profiled_steps"]) == 5
        for row in idle["profiled_steps"]:
            print(
                "[idle] step={step} wall_ms={wall_ms} gpu_kernel_sum_ms={gpu_kernel_sum_ms} "
                "idle_fraction={gpu_idle_fraction} cuda_launches={cuda_launches} "
                "cpu_launch_ms={cpu_launch_api_ms} cpu_ms_per_launch={cpu_ms_per_launch_api}".format(
                    **row
                ),
                flush=True,
            )
        print("[idle] top kernels (steps 10-14):", flush=True)
        for item in idle["top_kernels"]:
            print(f"[idle] kernel={item['name']} total_ms={item['time_ms']} count={item['count']}",
                  flush=True)
        print(f"[idle] json={out / 'idle.json'}", flush=True)
        if not profile_complete:
            print(f"[idle] expected 5 profiled steps, found {len(idle['profiled_steps'])}",
                  flush=True)
    else:
        print("[idle] profiler output missing; see startup_probe.json and solution_1/train.log",
              flush=True)

    startup = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("t4_startup.py")), "--run-dir", str(out)],
        cwd=repo,
        check=False,
    )
    if startup.returncode != 0:
        sys.exit(f"[idle] startup summary failed with exit code {startup.returncode}")
    if not profile_complete:
        sys.exit("[idle] training finished, but the 10-14 step profile was incomplete")


if __name__ == "__main__":
    main()
