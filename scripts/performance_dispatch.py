"""Measure the actual PEFT-to-packed-model dispatch and its candidate variants."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def clean_env():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SOUP_X_") or key == "SOUP_RECOMPUTE_MLP":
            env.pop(key)
    return env


status = {}
for group, files in {
    "dispatch": ["solution/test_layer_tail_dispatch.py"],
    "lora": ["solution/test_lora_output_storage.py"],
    "metrics": ["solution/test_metric_snapshot.py"],
    "codec": ["solution/test_activation_codec.py", "solution/test_norm_storage.py"],
}.items():
    status[group] = subprocess.call(
        [sys.executable, "-m", "pytest", *files, "-q", "-x", "-p", "no:cacheprovider"],
        env=clean_env(),
    )
print("DISPATCH_TESTS", json.dumps(status), flush=True)
Path("/kaggle/working/dispatch-tests.json").write_text(json.dumps(status, indent=2))
common = dict(SOUP_X_ACT_INT8="1", SOUP_X_NORM_SAVE16="1", SOUP_X_FPRINT="1",
              SOUP_X_MERGE_VARIANTS="1", SOUP_X_TRACE_DISPATCH="1")
if status["metrics"] == 0:
    common["SOUP_X_METRICS_BATCHED"] = "1"
varlen = dict(common, SOUP_X_VARLEN="1")
four = dict(varlen, SOUP_X_ACT_INT4="1")
four.pop("SOUP_X_ACT_INT8")
cases = [("packed_int8", common), ("varlen_int8", varlen), ("varlen_int4", four)]
if status["lora"] == 0:
    cases.append(("varlen_int4_lora", dict(four, SOUP_X_LINEAR_OUT="1", SOUP_X_LORA="1")))
rows = []
for name, flags in cases:
    env = clean_env()
    env.update(flags)
    out = ROOT / "runs_dispatch" / name
    print("DISPATCH_START", name, flags, flush=True)
    code = subprocess.call([sys.executable, "-u", "harness/run_task.py", "--only", "solution",
                            "--out", str(out)], env=env)
    row = dict(name=name, flags=flags, status=code)
    if (out / "result.json").exists():
        result = json.loads((out / "result.json").read_text())
        row.update(result.get("summary", {}).get("solution", {}))
        row["metrics"] = [r.get("metrics") for r in result.get("runs", {}).get("solution", [])]
    for log in out.glob("solution_*/train.log"):
        for line in log.read_text().splitlines():
            if "SOUP_LAYER_TAIL" in line:
                print(line, flush=True)
        if not row.get("ok"):
            print(log.read_text()[-16000:], flush=True)
    rows.append(row)
    print("DISPATCH_RESULT", json.dumps(row), flush=True)
    Path("/kaggle/working/dispatch-summary.json").write_text(json.dumps(rows, indent=2))

# Keep the already allocated T4 for full qualification after the quick screen.
# This is notebook tooling, outside every measured training child process.
qualified = []
eligible = [row for row in rows if row.get("ok")]
if status["dispatch"] == 0 and status["codec"] == 0 and eligible:
    fastest = min(eligible, key=lambda row: row["wall_s"])
    smallest = min(eligible, key=lambda row: (row["gpu_peak_mib"], row["wall_s"]))
    shortlist = {row["name"]: row for row in (fastest, smallest)}
    for name, candidate in shortlist.items():
        env = clean_env()
        env.update(candidate["flags"])
        out = ROOT / "runs_dispatch_qualification" / name
        print("DISPATCH_QUALIFY_START", name, candidate["flags"], flush=True)
        code = subprocess.call([sys.executable, "-u", "harness/run_task.py", "--repeats", "2",
                                "--seed", "2026", "--out", str(out)], env=env)
        check = subprocess.call([sys.executable, "-u", "harness/check_rules.py", "--result",
                                 str(out / "result.json")], env=env)
        row = dict(name=name, flags=candidate["flags"], run_status=code, check_status=check)
        for filename in ("result.json", "checks.json"):
            source = out / filename
            if source.exists():
                value = json.loads(source.read_text())
                row["summary" if filename == "result.json" else "checks"] = (
                    value.get("summary") if filename == "result.json" else value)
                (Path("/kaggle/working") / (name + "-" + filename)).write_bytes(source.read_bytes())
        qualified.append(row)
        print("DISPATCH_QUALIFY_RESULT", json.dumps(row), flush=True)
        Path("/kaggle/working/dispatch-qualification-summary.json").write_text(
            json.dumps(qualified, indent=2))
with zipfile.ZipFile("/kaggle/working/dispatch-evidence.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for directory in ("runs_dispatch", "runs_dispatch_qualification"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".yaml", ".log"):
                archive.write(path, str(path.relative_to(ROOT)))
print("DISPATCH_COMPLETE", flush=True)
