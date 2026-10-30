"""Does Triton use tensor cores for tl.dot on this GPU? Compare a plain Triton fp16 matmul with
torch.mm (cuBLAS) and count mma / fma instructions in the generated PTX. Reads nothing, writes nothing."""
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * sam + (k + rk)[None, :] * sak,
                    mask=(rm[:, None] < M) & ((k + rk)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k + rk)[:, None] * sbk + rn[None, :] * sbn,
                    mask=((k + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc.to(tl.float16),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    print("torch", torch.__version__, "triton", triton.__version__, "device", torch.cuda.get_device_name(0),
          "capability", torch.cuda.get_device_capability(0))
    for M, N, K in ((2048, 1536, 1536), (2048, 8960, 1536)):
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        c = torch.empty(M, N, device="cuda", dtype=torch.float16)
        flops = 2 * M * N * K
        t_ref = bench(lambda: torch.mm(a, b))
        print(f"\nM={M} N={N} K={K}: torch.mm {t_ref:.3f} ms = {flops / t_ref / 1e9:.1f} TFLOP/s")
        for BM, BN, BK, warps in ((64, 64, 32, 4), (128, 128, 32, 8)):
            grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
            run = lambda: matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                                              b.stride(1), c.stride(0), c.stride(1),
                                              BM=BM, BN=BN, BK=BK, num_warps=warps)
            try:
                compiled = run()
                t = bench(run)
                err = (c.float() - torch.mm(a, b).float()).abs().max().item()
                ptx = compiled.asm["ptx"]
                print(f"  triton {BM}x{BN}x{BK} warps={warps}: {t:.3f} ms = {flops / t / 1e9:.1f} TFLOP/s "
                      f"({t / t_ref:.1f}x slower than torch.mm), max err {err:.3f}, "
                      f"PTX mma={ptx.count('mma.')} fma.f32={ptx.count('fma.rn.f32')}")
            except Exception as exc:  # noqa: BLE001
                print(f"  triton {BM}x{BN}x{BK}: FAILED {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()
