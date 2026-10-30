"""Measure startup imports and first-use Triton compilation on a CUDA GPU."""

import os
import tempfile
import time


def timed_call(torch, name, shape, call):
    durations = []
    for _ in range(2):
        torch.cuda.synchronize()
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)

    delta = durations[0] - durations[1]
    print(
        f"{name:38s} {shape:18s} first={durations[0]:.4f}s "
        f"second={durations[1]:.4f}s first-minus-second={delta:.4f}s",
        flush=True,
    )
    return durations[0], durations[1], max(delta, 0.0)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with tempfile.TemporaryDirectory(prefix="soup-startup-triton-") as cache_dir:
        os.environ["TRITON_CACHE_DIR"] = cache_dir
        print(f"Fresh Triton cache: {cache_dir}", flush=True)

        started = time.perf_counter()
        import torch

        torch_import_s = time.perf_counter() - started

        if not torch.cuda.is_available():
            raise SystemExit("A CUDA GPU is required; run this on the Tesla T4.")

        started = time.perf_counter()
        import transformers
        import trl
        import peft
        import bitsandbytes

        training_import_s = time.perf_counter() - started

        started = time.perf_counter()
        import solution

        solution.apply()
        solution_import_apply_s = time.perf_counter() - started

        import triton
        from solution.add_rmsnorm import add_rmsnorm_bwd_kernel, add_rmsnorm_fwd_kernel
        from solution.ce_loss import BLOCK_V, head_stats_kernel
        from solution.nf4_dequant import nf4_dequant_kernel
        from solution.rmsnorm import rmsnorm_bwd_kernel, rmsnorm_fwd_kernel
        from solution.rope import rope_kernel

        started = time.perf_counter()
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        cuda_init_s = time.perf_counter() - started

        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"import torch: {torch_import_s:.3f}s", flush=True)
        print(f"import transformers/trl/peft/bitsandbytes: {training_import_s:.3f}s", flush=True)
        print(f"import solution + apply(): {solution_import_apply_s:.3f}s", flush=True)
        print(f"CUDA context initialization: {cuda_init_s:.3f}s", flush=True)
        print("Kernel timings (first call includes compilation; difference is an estimate):",
              flush=True)

        compile_estimates = []
        batch = 4
        lengths = (200, 311, 512)
        width = 1536
        block_width = triton.next_power_of_2(width)
        max_rows = batch * max(lengths)

        x = torch.zeros((max_rows, width), device="cuda", dtype=torch.float32)
        weight = torch.ones(width, device="cuda", dtype=torch.float32)
        y = torch.empty((max_rows, width), device="cuda", dtype=torch.float16)
        rstd = torch.empty(max_rows, device="cuda", dtype=torch.float32)
        dy = torch.ones_like(y)
        dx = torch.empty_like(x)

        residual = torch.zeros_like(x)
        delta = torch.zeros_like(y)
        total = torch.empty_like(x)
        add_y = torch.empty_like(y)
        dsum = torch.zeros_like(x)
        dres = torch.empty_like(x)
        ddelta = torch.empty_like(y)

        for seq in lengths:
            rows = batch * seq
            shape = f"batch={batch}, T={seq}, H={width}"
            compile_estimates.append(timed_call(
                torch, "rmsnorm_fwd_kernel", shape,
                lambda rows=rows: rmsnorm_fwd_kernel[(rows,)](
                    x[:rows], weight, y[:rows], rstd[:rows], width, 1e-6,
                    BLOCK=block_width, num_warps=8)))
            compile_estimates.append(timed_call(
                torch, "rmsnorm_bwd_kernel", shape,
                lambda rows=rows: rmsnorm_bwd_kernel[(rows,)](
                    dy[:rows], x[:rows], weight, rstd[:rows], dx[:rows], width,
                    BLOCK=block_width, num_warps=8)))

            compile_estimates.append(timed_call(
                torch, "add_rmsnorm_fwd_kernel", shape,
                lambda rows=rows: add_rmsnorm_fwd_kernel[(rows,)](
                    residual[:rows], delta[:rows], weight, total[:rows], add_y[:rows],
                    rstd[:rows], width, 1e-6, BLOCK=block_width, num_warps=8)))
            compile_estimates.append(timed_call(
                torch, "add_rmsnorm_bwd_kernel HAS_DSUM=True", shape,
                lambda rows=rows: add_rmsnorm_bwd_kernel[(rows,)](
                    dy[:rows], dsum[:rows], total[:rows], weight, rstd[:rows],
                    dres[:rows], ddelta[:rows], width, BLOCK=block_width,
                    HAS_DSUM=True, num_warps=8)))
            if seq == lengths[0]:
                compile_estimates.append(timed_call(
                    torch, "add_rmsnorm_bwd_kernel HAS_DSUM=False", shape,
                    lambda rows=rows: add_rmsnorm_bwd_kernel[(rows,)](
                        dy[:rows], dy[:rows], total[:rows], weight, rstd[:rows],
                        dres[:rows], ddelta[:rows], width, BLOCK=block_width,
                        HAS_DSUM=False, num_warps=8)))

            q_base = torch.zeros((batch, seq, 12, 128), device="cuda", dtype=torch.float16)
            k_base = torch.zeros((batch, seq, 2, 128), device="cuda", dtype=torch.float16)
            cos = torch.ones((1, seq, 128), device="cuda", dtype=torch.float32)
            sin = torch.zeros_like(cos)
            for label, source in (("q", q_base.transpose(1, 2)),
                                  ("k", k_base.transpose(1, 2))):
                out = torch.empty_like(source)
                for backward in (False, True):
                    direction = "bwd" if backward else "fwd"

                    def launch_rope(source=source, out=out, backward=backward):
                        rope_kernel[(seq, source.shape[1], batch)](
                            source, cos, sin, out, source.stride(0), source.stride(1),
                            source.stride(2), out.stride(0), out.stride(1), out.stride(2),
                            0, cos.stride(1), HALF=64, BACKWARD=backward, num_warps=2)

                    compile_estimates.append(timed_call(
                        torch, f"rope_kernel {label} {direction}", shape, launch_rope))

            vocab = 151936
            rows = batch * seq % 256 or 256
            logits = torch.zeros((rows, vocab), device="cuda", dtype=torch.float16)
            targets = torch.zeros(rows, device="cuda", dtype=torch.int32)
            losses = torch.empty(rows, device="cuda", dtype=torch.float32)
            entropies = torch.empty_like(losses)
            hits = torch.empty(rows, device="cuda", dtype=torch.int32)
            for write_grad in (False, True):
                target = targets if write_grad else torch.full_like(targets, -100)

                def launch_head(write_grad=write_grad, target=target):
                    head_stats_kernel[(rows,)](
                        logits, target, losses, entropies, hits, vocab, vocab,
                        BLOCK=BLOCK_V, WRITE_GRAD=write_grad, num_warps=8)

                compile_estimates.append(timed_call(
                    torch, f"head_stats_kernel WRITE_GRAD={write_grad}",
                    f"T={seq}, rows={rows}, V={vocab}", launch_head))

            del q_base, k_base, cos, sin, out, source, launch_rope, logits, targets
            del losses, entropies, hits, launch_head

        nf4_shape = (8960, 1536)
        n_blocks = nf4_shape[0] * nf4_shape[1] // 64
        packed = torch.zeros(n_blocks * 32, device="cuda", dtype=torch.uint8)
        absmax_q = torch.zeros(n_blocks, device="cuda", dtype=torch.uint8)
        code2 = torch.ones(256, device="cuda", dtype=torch.float32)
        absmax2 = torch.ones(triton.cdiv(n_blocks, 256), device="cuda", dtype=torch.float32)
        offset = torch.zeros(1, device="cuda", dtype=torch.float32)
        nf4_code = torch.zeros(16, device="cuda", dtype=torch.float32)
        dequantized = torch.empty(nf4_shape, device="cuda", dtype=torch.float16)

        def launch_nf4():
            nf4_dequant_kernel[(triton.cdiv(n_blocks, 32),)](
                packed, absmax_q, code2, absmax2, offset, nf4_code, dequantized,
                n_blocks, BLOCK_SIZE=64, BLOCKS=32, SECOND_BLOCK=256, num_warps=4)

        compile_estimates.append(timed_call(
            torch, "nf4_dequant_kernel", "weight=8960x1536, block=64", launch_nf4))

        torch.cuda.synchronize()
        first_total = sum(first for first, _second, _delta in compile_estimates)
        second_total = sum(second for _first, second, _delta in compile_estimates)
        compile_total = sum(delta for _first, _second, delta in compile_estimates)
        print(f"Total cold calls: {first_total:.3f}s", flush=True)
        print(f"Total warm calls: {second_total:.3f}s", flush=True)
        print(f"Total first-minus-second estimate: {compile_total:.3f}s", flush=True)


if __name__ == "__main__":
    main()
