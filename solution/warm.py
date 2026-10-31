"""Compile our Triton kernels while Soup is still loading the model.

Cold compilation of our kernels costs about 4.7 s inside the first training step. `JITFunction.warmup`
is Triton's own API for compiling a kernel without launching it: no buffers are allocated, no kernel
runs, and the harness skips such calls when it counts launches (`kwargs.get("warmup")`). This module
calls it from one daemon thread for the dtype / divisibility variants the run is expected to use, so
the compiled kernels are already in Triton's in-memory cache when the first step needs them. A variant
we guess wrong costs nothing but a compile in the thread; the training step then compiles it itself.

Off unless SOUP_X_WARM=1. SOUP_X_WARM_LOG=1 prints every Triton compile with the thread that made it,
so a run shows which variants the warm-up covered and which the main thread still compiled.
"""

import os
import threading
import time

_T0 = time.time()
_started = False


def _log(message):
    if os.environ.get("SOUP_X_WARM_LOG"):
        print(f"[warm] {time.time() - _T0:7.2f}s {threading.current_thread().name}: {message}",
              flush=True)


def _install_compile_log():
    if not os.environ.get("SOUP_X_WARM_LOG"):
        return

    def hook(*args, **kwargs):
        fn = kwargs.get("fn")
        name = getattr(getattr(fn, "fn", fn), "__name__", "?")
        _log(f"compiled {name} manual_warmup={kwargs.get('is_manual_warmup')} "
             f"already={kwargs.get('already_compiled')}")

    try:
        from triton import knobs
        knobs.runtime.jit_post_compile_hook = hook
    except Exception as exc:  # noqa: BLE001 - the log is optional
        _log(f"no compile hook: {exc!r}")


def _variants():
    """(kernel, positional args, constexpr kwargs) tuples. Pointers are torch dtypes, which
    `warmup` turns into aligned mock tensors; ints and floats are plain values."""
    import torch

    from .add_rmsnorm import add_rmsnorm_bwd_kernel, add_rmsnorm_fwd_kernel
    from .ce_loss import BLOCK_V, head_stats_kernel
    from .nf4_dequant import BLOCKS, nf4_dequant_kernel
    from .rmsnorm import rmsnorm_bwd_kernel, rmsnorm_fwd_kernel
    from .rope import rope_kernel

    f16, f32, i32, u8 = torch.float16, torch.float32, torch.int32, torch.uint8
    width, vocab, eps, block = 1536, 151936, 1e-6, 2048
    out = []

    # RMSNorm: fp32 residual stream in, fp16 (autocast) out; norm weights are fp32 after kbit prep.
    for x in (f32, f16):
        out.append((rmsnorm_fwd_kernel, (x, f32, f16, f32, width, eps), dict(BLOCK=block, num_warps=8)))
        out.append((rmsnorm_bwd_kernel, (f16, x, f32, f32, x, width), dict(BLOCK=block, num_warps=8)))

    # Residual add + RMSNorm: fp32 residual, fp16 branch output.
    for res, delta in ((f32, f16), (f16, f16)):
        total = f32 if res is f32 else f16
        out.append((add_rmsnorm_fwd_kernel, (res, delta, f32, total, f16, f32, width, eps),
                    dict(BLOCK=block, num_warps=8)))
        for has_dsum in (True, False):
            out.append((add_rmsnorm_bwd_kernel,
                        (f16, total if has_dsum else f16, total, f32, f32, res, delta, width),
                        dict(BLOCK=block, HAS_DSUM=has_dsum, num_warps=8)))

    # RoPE: seq divisible by 16 or not (the batch is padded to its longest row), cos/sin fp32 or fp16.
    for cos in (f32, f16):
        for backward in (False, True):
            for seq in (64, 63):
                out.append((rope_kernel,
                            (f16, cos, cos, f16, seq, seq * 1536, 128, 1536, seq * 1536, 128, 1536,
                             seq * 128, 128),
                            dict(HALF=64, BACKWARD=backward, num_warps=2)))

    # Chunked head statistics, with and without the gradient.
    for write_grad in (True, False):
        out.append((head_stats_kernel, (f16, i32, f32, f32, i32, vocab, vocab),
                    dict(BLOCK=BLOCK_V, WRITE_GRAD=write_grad, num_warps=8)))

    # NF4 dequantization (nested): every weight shape has n_blocks divisible by 16.
    out.append((nf4_dequant_kernel, (u8, u8, f32, f32, f32, f32, f16, 36864),
                dict(BLOCK_SIZE=64, BLOCKS=BLOCKS, SECOND_BLOCK=256, num_warps=4)))
    return out


def _run():
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.cuda.init()
        count = 0
        for kernel, args, kwargs in _variants():
            try:
                kernel.warmup(*args, grid=(1,), **kwargs)
                count += 1
            except Exception as exc:  # noqa: BLE001 - a wrong guess must not hurt the run
                _log(f"skipped {getattr(kernel.fn, '__name__', '?')}: {exc!r}")
        _log(f"warm-up finished: {count} variants")
    except Exception as exc:  # noqa: BLE001
        _log(f"warm-up failed: {exc!r}")


def start_warm_thread():
    global _started
    if _started:
        return
    _install_compile_log()  # the log also works with the warm-up off, to compare runs
    if os.environ.get("SOUP_X_WARM") != "1":
        return
    _started = True
    threading.Thread(target=_run, name="triton-warm", daemon=True).start()
