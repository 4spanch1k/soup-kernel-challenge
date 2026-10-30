"""Compare one PEFT-style LoRA call with the fused path on a CUDA GPU."""

import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from solution.lora_fuse import lora_fused

WARMUP = 5
REPEATS = 20
MIB = 1024**2


def _reference(x, base, a, b, scaling, p):
    dropped = F.dropout(x.to(a.dtype), p=p, training=True)
    lora_out = F.linear(F.linear(dropped, a), b) * scaling
    return (base + lora_out).to(base.dtype)


def _measure(function, x, base, a, b, scaling, p):
    def step():
        x.grad = base.grad = a.grad = b.grad = None
        output = function(x, base, a, b, scaling, p)
        output.float().mean().backward()
        return output

    for _ in range(WARMUP):
        output = step()
        del output
    x.grad = base.grad = a.grad = b.grad = None
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = step()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        del output
    peak = torch.cuda.max_memory_allocated()
    return statistics.median(times), peak / MIB, (peak - baseline) / MIB


def main():
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA GPU (T4 recommended).")
    print(f"GPU: {torch.cuda.get_device_name()}; median over {REPEATS} forward+backward calls")
    for n in (826, 2048):
        for name, out_features in (("q_proj", 1536), ("v_proj", 256)):
            x = torch.randn(n, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
            base = torch.randn(n, out_features, device="cuda", dtype=torch.float16,
                               requires_grad=True)
            a = torch.randn(16, 1536, device="cuda", dtype=torch.float32, requires_grad=True)
            b = torch.randn(out_features, 16, device="cuda", dtype=torch.float32,
                            requires_grad=True)
            for label, function in (("PEFT", _reference), ("fused", lora_fused)):
                with torch.autocast("cuda", dtype=torch.float16):
                    median_ms, peak_mib, extra_mib = _measure(
                        function, x, base, a, b, 2.0, 0.05)
                print(f"{name:7s} N={n:4d} {label:5s}: {median_ms:7.3f} ms, "
                      f"peak={peak_mib:7.1f} MiB, extra={extra_mib:6.1f} MiB")
            del x, base, a, b
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
