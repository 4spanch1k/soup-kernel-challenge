"""T4 bake-off: dequantize + cuBLAS versus fused Triton NF4 matrix multiplies."""

import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bitsandbytes.functional as bnb_functional

from solution.fused4bit import fused_matmul_forward, fused_matmul_input_grad
from solution.nf4_dequant import nf4_dequantize

M_VALUES = (400, 826, 1400, 2048)
SHAPES = (
    ("square", 1536, 1536),
    ("kv", 256, 1536),
    ("gate_up", 8960, 1536),
    ("down", 1536, 8960),
)
WARMUP = 5
ITERATIONS = 40


def _time_first_call(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def _median_cuda_ms(function):
    for _ in range(WARMUP):
        function()
    torch.cuda.synchronize()

    events = []
    for _ in range(ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in events)


def _errors(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-5)
    return difference.max().item(), relative.max().item()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA GPU; use a Tesla T4 for the bake-off.")

    device = torch.cuda.get_device_name()
    print(f"GPU: {device}; warmup={WARMUP}; event samples={ITERATIONS}")
    print("Building NF4 weights with bitsandbytes (nested statistics, blocksize 64).")
    records = []
    cold_calls = []

    for name, n, k in SHAPES:
        torch.manual_seed(0)
        source_weight = (torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.02)
        packed, state = bnb_functional.quantize_4bit(
            source_weight, blocksize=64, compress_statistics=True, quant_type="nf4")

        for m in M_VALUES:
            x = torch.randn((m, k), device="cuda", dtype=torch.float16)
            dy = torch.randn((m, n), device="cuda", dtype=torch.float16)
            forward_ref = lambda: F.linear(x, nf4_dequantize(packed, state))
            backward_ref = lambda: torch.matmul(dy, nf4_dequantize(packed, state))
            forward_fused = lambda: fused_matmul_forward(x, packed, state)
            backward_fused = lambda: fused_matmul_input_grad(dy, packed, state)

            forward, forward_cold_ms = _time_first_call(forward_fused)
            backward, backward_cold_ms = _time_first_call(backward_fused)
            cold_calls.append((name, m, "forward", forward_cold_ms))
            cold_calls.append((name, m, "backward", backward_cold_ms))

            expected_forward = forward_ref()
            expected_backward = backward_ref()
            torch.cuda.synchronize()
            fwd_abs, fwd_rel = _errors(forward, expected_forward)
            bwd_abs, bwd_rel = _errors(backward, expected_backward)

            for direction, reference, fused, max_abs, max_rel in (
                    ("forward", forward_ref, forward_fused, fwd_abs, fwd_rel),
                    ("backward", backward_ref, backward_fused, bwd_abs, bwd_rel)):
                reference_ms = _median_cuda_ms(reference)
                fused_ms = _median_cuda_ms(fused)
                records.append({
                    "name": name,
                    "shape": (n, k),
                    "m": m,
                    "direction": direction,
                    "reference_ms": reference_ms,
                    "fused_ms": fused_ms,
                    "speedup": reference_ms / fused_ms,
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                })

    print("\nFirst-call wall time per case (compilation on cache miss, plus launch):")
    print("class       M     direction   first_call_ms")
    for name, m, direction, elapsed in cold_calls:
        print(f"{name:<11} {m:<5} {direction:<11} {elapsed:>10.3f}")

    print("\nCUDA-event median timings; speedup = reference / fused:")
    print("class       [N,K]          M     direction   A_ms     B_ms     speedup   max_abs   max_rel")
    for row in records:
        n, k = row["shape"]
        print(f"{row['name']:<11} [{n:>4},{k:<4}] {row['m']:<5} "
              f"{row['direction']:<11} {row['reference_ms']:>7.3f} "
              f"{row['fused_ms']:>7.3f} {row['speedup']:>8.3f} "
              f"{row['max_abs']:>9.4g} {row['max_rel']:>9.4g}")

    print("\nVerdict by shape class (median speedup over M and both directions):")
    for name, _, _ in SHAPES:
        speedups = [row["speedup"] for row in records if row["name"] == name]
        median = statistics.median(speedups)
        verdict = "fused wins" if median > 1.05 else "reference wins" if median < 0.95 else "close"
        print(f"{name:<11} {median:.3f}x  {verdict}")


if __name__ == "__main__":
    main()
