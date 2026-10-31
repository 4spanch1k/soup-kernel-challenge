"""GEMM micro-benchmark for the Soup Kernel Challenge shapes (plain PyTorch, no repo code, no downloads).

Run on a Colab T4:   !python gemm_bench.py            (all sections, ~8-12 min)
                     !python gemm_bench.py --sections head,pad --Ms 1170
Sections: chain, head, layer, pad, sweep, layout, concat, streams, knobs
Every number is ms per call (CUDA events, median of 3 timed loops) and TFLOP/s = 2*M*N*K / time.
The GPU clock / power is printed after each section (a T4 is power-capped at 70 W: clocks fall as it heats).
"""
import argparse
import math
import statistics
import subprocess

import torch
import torch.nn.functional as F

H, I, KV, V = 1536, 8960, 256, 151936      # hidden, intermediate, k/v width, vocab
SM = 40                                    # T4 streaming multiprocessors
dev, dt = "cuda", torch.float16


def mk(*shape):
    t = torch.empty(*shape, device=dev, dtype=dt)
    t.normal_(0, 0.02)
    return t


def rot(make, n=4):
    """n distinct buffers used round-robin (weights of different layers are different in the real step)."""
    bufs = [make() for _ in range(n)]
    i = [0]

    def nxt():
        i[0] = (i[0] + 1) % n
        return bufs[i[0]]
    return nxt


def timeit(fn, flops=None, min_ms=200.0, reps=3):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); fn(); b.record(); b.synchronize()
    n = max(5, int(min_ms / max(a.elapsed_time(b), 0.02)))
    out = []
    for _ in range(reps):
        a.record()
        for _ in range(n):
            fn()
        b.record(); b.synchronize()
        out.append(a.elapsed_time(b) / n)
    ms = statistics.median(out)
    return ms, (flops / ms / 1e9 if flops else float("nan"))


def gpu_state():
    try:
        q = "clocks.sm,clocks.max.sm,power.draw,temperature.gpu"
        s = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        return s
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e}"


def model_E(N, M, tm=256, tn=128, occ=1):
    """Tile x wave efficiency of cuBLAS column-major C[N x M] = W[N x K] X^T, tile tm (along N) x tn (along M).
    ASSUMPTION: 256x128 means 256 along the out-feature dim and 128 along the token dim. The sweep section tests it."""
    tiles = math.ceil(N / tm) * math.ceil(M / tn)
    waves = math.ceil(tiles / (SM * occ))
    return (N * M) / (math.ceil(N / tm) * tm * math.ceil(M / tn) * tn) * tiles / (waves * SM * occ)


def row(name, ms, tf, extra=""):
    print(f"  {name:<44s} {ms:8.3f} ms  {tf:6.1f} TFLOP/s  {extra}")


# ---------------------------------------------------------------- layer: every GEMM of one decoder layer
def sec_layer(Ms):
    print("\n== layer: fwd = F.linear (cuBLAS 'tn'), dX = grad @ W (cuBLAS 'nn'); bias on q,k,v goes through addmm/cuBLASLt")
    for M in Ms:
        print(f" M={M}")
        for name, N, K, bias in [("q  fwd  N=1536 K=1536 +bias", H, H, 1), ("k  fwd  N=256  K=1536 +bias", KV, H, 1),
                                 ("v  fwd  N=256  K=1536 +bias", KV, H, 1), ("o  fwd  N=1536 K=1536", H, H, 0),
                                 ("gate/up fwd N=8960 K=1536", I, H, 0), ("down fwd N=1536 K=8960", H, I, 0)]:
            w = rot(lambda: mk(N, K)); x = mk(M, K); b = mk(N) if bias else None
            ms, tf = timeit(lambda: F.linear(x, w(), b), 2 * M * N * K)
            row(name, ms, tf, f"model E(256x128)={model_E(N, M):.2f}")
        for name, N, K in [("q/o dX  N=1536 K=1536", H, H), ("k/v dX  N=1536 K=256", H, KV),
                           ("gate/up dX N=1536 K=8960", H, I), ("down dX N=8960 K=1536", I, H)]:
            w = rot(lambda: mk(K, N)); g = mk(M, K)
            ms, tf = timeit(lambda: torch.matmul(g, w()), 2 * M * N * K)
            row(name, ms, tf, f"model E(256x128)={model_E(N, M):.2f}")
    print("  gpu:", gpu_state())


# ---------------------------------------------------------------- pad: zero rows up to a multiple
def sec_pad(Ms):
    print("\n== pad: same GEMM with M rounded UP (zero rows). d = padded - base in ms (negative = faster).")
    shapes = [("gate fwd (tn) N=8960", "f", I, H), ("o fwd (tn) N=1536", "f", H, H), ("down fwd (tn) N=1536 K=8960", "f", H, I),
              ("gate dX (nn) N=1536 K=8960", "b", H, I), ("down dX (nn) N=8960 K=1536", "b", I, H)]
    for M in Ms:
        print(f" base M={M}")
        for name, kind, N, K in shapes:
            if kind == "f":
                w = rot(lambda: mk(N, K)); call = lambda x, w=w: F.linear(x, w())
            else:
                w = rot(lambda: mk(K, N)); call = lambda x, w=w: torch.matmul(x, w())
            kk = K if kind == "f" else K
            base = None
            line = []
            for tag, m in [("M", M), ("+1", M + 1), ("up8", -(-M // 8) * 8), ("up64", -(-M // 64) * 64),
                           ("up128", -(-M // 128) * 128), ("up256", -(-M // 256) * 256)]:
                x = mk(m, kk)
                ms, _ = timeit(lambda: call(x), None, min_ms=120)
                base = ms if base is None else base
                line.append(f"{tag}({m}):{ms:.3f}({ms - base:+.3f})")
            print(f"  {name:<30s}", " ".join(line))
    print("  gpu:", gpu_state())


# ---------------------------------------------------------------- sweep: staircase in M
def sec_sweep(_):
    print("\n== sweep: ms vs M. A staircase with step 128 (or 256) in M shows the token-dimension tile size.")
    for name, N, K in [("gate fwd N=8960 K=1536", I, H), ("o fwd N=1536 K=1536", H, H)]:
        w = mk(N, K)
        pts = []
        for m in list(range(1016, 1296, 8)):
            x = mk(m, K)
            ms, _ = timeit(lambda: F.linear(x, w), None, min_ms=60, reps=2)
            pts.append((m, ms))
        print(f"  {name}: " + " ".join(f"{m}:{ms:.3f}" for m, ms in pts))
    print("  gpu:", gpu_state())


# ---------------------------------------------------------------- concat: q|k|v and gate|up
def sec_concat(Ms):
    print("\n== concat: separate GEMMs (as in the layer) vs one GEMM on the stacked weight (no extra dequant: the dequant kernel can write slices)")
    for M in Ms:
        print(f" M={M}")
        wq, wk, wv = (rot(lambda n=n: mk(n, H), 2) for n in (H, KV, KV))
        bq, bk, bv = mk(H), mk(KV), mk(KV)
        wqkv = rot(lambda: mk(H + 2 * KV, H), 2); bqkv = mk(H + 2 * KV)
        x = mk(M, H)
        fl = 2 * M * (H + 2 * KV) * H
        ms_s, tf = timeit(lambda: (F.linear(x, wq(), bq), F.linear(x, wk(), bk), F.linear(x, wv(), bv)), fl)
        row("qkv fwd separate (3 calls)", ms_s, tf)
        ms_f, tf = timeit(lambda: F.linear(x, wqkv(), bqkv), fl)
        row("qkv fwd fused   (1 call, N=2048)", ms_f, tf, f"saves {ms_s - ms_f:+.3f} ms/layer = {28 * (ms_s - ms_f):+.1f} ms/step")
        gq, gk, gv = mk(M, H), mk(M, KV), mk(M, KV)
        wq2, wk2, wv2 = (rot(lambda n=n: mk(n, H), 2) for n in (H, KV, KV))
        ms_s, tf = timeit(lambda: torch.matmul(gq, wq2()) + torch.matmul(gk, wk2()) + torch.matmul(gv, wv2()), fl)
        row("qkv dX separate (3 mm + 2 adds)", ms_s, tf)
        wf = rot(lambda: mk(H + 2 * KV, H), 2)
        ms_c, tf = timeit(lambda: torch.matmul(torch.cat([gq, gk, gv], 1), wf()), fl)
        row("qkv dX fused incl. torch.cat of grads", ms_c, tf, f"saves {ms_s - ms_c:+.3f} ms/layer = {28 * (ms_s - ms_c):+.1f} ms/step")
        gcat = mk(M, H + 2 * KV)
        ms_n, tf = timeit(lambda: torch.matmul(gcat, wf()), fl)
        row("qkv dX fused, grads already contiguous", ms_n, tf)
        wg, wu = rot(lambda: mk(I, H), 2), rot(lambda: mk(I, H), 2)
        wgu = rot(lambda: mk(2 * I, H), 2)
        fl2 = 2 * M * 2 * I * H
        ms_s, tf = timeit(lambda: (F.linear(x, wg()), F.linear(x, wu())), fl2)
        row("gate,up fwd separate (2 calls)", ms_s, tf)
        ms_f, tf = timeit(lambda: F.linear(x, wgu()), fl2)
        row("gate|up fwd fused (N=17920)", ms_f, tf, f"saves {ms_s - ms_f:+.3f} ms/layer = {28 * (ms_s - ms_f):+.1f} ms/step")
        gg, gu = mk(M, I), mk(M, I)
        wg2, wu2 = rot(lambda: mk(I, H), 2), rot(lambda: mk(I, H), 2)
        ms_s, tf = timeit(lambda: torch.matmul(gg, wg2()) + torch.matmul(gu, wu2()), fl2)
        row("gate,up dX separate (2 mm + add)", ms_s, tf)
        gc = mk(M, 2 * I); wf2 = rot(lambda: mk(2 * I, H), 2)
        ms_n, tf = timeit(lambda: torch.matmul(gc, wf2()), fl2)
        row("gate|up dX fused, grads contiguous", ms_n, tf, f"saves {ms_s - ms_n:+.3f} ms/layer = {28 * (ms_s - ms_n):+.1f} ms/step")
    print("  gpu:", gpu_state())


# ---------------------------------------------------------------- streams: q,k,v concurrently
def sec_streams(Ms):
    print("\n== streams: q,k,v forward GEMMs on 3 CUDA streams vs sequential (tail-wave filling)")
    for M in Ms:
        wq, wk, wv = mk(H, H), mk(KV, H), mk(KV, H)
        bq, bk, bv = mk(H), mk(KV), mk(KV)
        x = mk(M, H)
        side = [torch.cuda.Stream() for _ in range(2)]

        def seq():
            return F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv)

        def par():
            cur = torch.cuda.current_stream()
            for s in side:
                s.wait_stream(cur)
            q = F.linear(x, wq, bq)
            with torch.cuda.stream(side[0]):
                k = F.linear(x, wk, bk)
            with torch.cuda.stream(side[1]):
                v = F.linear(x, wv, bv)
            cur.wait_stream(side[0]); cur.wait_stream(side[1])
            return q, k, v
        ms_s, _ = timeit(seq, None); ms_p, _ = timeit(par, None)
        print(f"  M={M}: sequential {ms_s:.3f} ms, 3 streams {ms_p:.3f} ms, saves {ms_s - ms_p:+.3f} ms/layer = {28 * (ms_s - ms_p):+.1f} ms/step")
    print("  gpu:", gpu_state())


# ---------------------------------------------------------------- knobs
def sec_knobs(Ms):
    print("\n== knobs: A-B-A on 4 shapes; B = the knob. Numerics of fp16 accumulation are checked against fp32.")
    M = Ms[len(Ms) // 2]
    cases = {}
    w1 = rot(lambda: mk(I, H)); x1 = mk(M, H); cases["gate fwd tn"] = (lambda: F.linear(x1, w1()), 2 * M * I * H)
    w2 = rot(lambda: mk(H, H)); x2 = mk(M, H); cases["o fwd tn"] = (lambda: F.linear(x2, w2()), 2 * M * H * H)
    w3 = rot(lambda: mk(I, H)); g3 = mk(M, I); cases["gate dX nn K=8960"] = (lambda: torch.matmul(g3, w3()), 2 * M * H * I)
    wh = mk(V, H); lg = mk(256, V); cases["head dX nn 256x151936x1536"] = (lambda: torch.mm(lg, wh), 2 * 256 * V * H)
    mm = torch.backends.cuda.matmul

    def knob(name, on, off):
        print(f" knob: {name}")
        try:
            for cname, (fn, fl) in cases.items():
                a, tfa = timeit(fn, fl, min_ms=150); on(); b, tfb = timeit(fn, fl, min_ms=150); off(); c, tfc = timeit(fn, fl, min_ms=150)
                print(f"  {cname:<30s} base {a:.3f}/{c:.3f} ms ({tfa:.1f}/{tfc:.1f} TF)   knob {b:.3f} ms ({tfb:.1f} TF)   ratio {b / ((a + c) / 2):.3f}")
        except Exception as e:  # noqa: BLE001
            off(); print("  not available:", repr(e))
    knob("preferred_blas_library('cublaslt')", lambda: torch.backends.cuda.preferred_blas_library("cublaslt"),
         lambda: torch.backends.cuda.preferred_blas_library("default"))
    knob("allow_fp16_reduced_precision_reduction=False", lambda: setattr(mm, "allow_fp16_reduced_precision_reduction", False),
         lambda: setattr(mm, "allow_fp16_reduced_precision_reduction", True))
    flag = next((n for n in ("allow_fp16_accumulation", "use_fp16_accumulation") if hasattr(mm, n)), None)
    if flag:
        knob(f"{flag}=True (changes numerics, NOT allowed for the submission; shown only to see if T4 gains)", lambda: setattr(mm, flag, True),
             lambda: setattr(mm, flag, False))
        a, b = mk(M, H), mk(I, H)
        ref = (a.float() @ b.float().t())
        base_err = (F.linear(a, b).float() - ref).abs().max().item()
        setattr(mm, flag, True); err = (F.linear(a, b).float() - ref).abs().max().item(); setattr(mm, flag, False)
        print(f"  max abs error vs fp32: default {base_err:.4g}, fp16 accumulation {err:.4g}")
    else:
        print(" fp16-accumulation flag not found in this torch")
    if hasattr(torch.backends.cuda, "cublas_workspace_size"):
        knob("cublas_workspace_size(64 MiB)", lambda: torch.backends.cuda.cublas_workspace_size(64 << 20),
             lambda: torch.backends.cuda.cublas_workspace_size(8 << 20))  # ~PyTorch's default (:4096:2:16:8)
    else:
        print(" cublas_workspace_size API not found in this torch")
    print("  gpu:", gpu_state())


# ---------------------------------------------------------------- head
def sec_head(_):
    print("\n== head: logits chunk = mm(rows[256,1536], W[V,1536].T); dX = mm(logits[:lab], W)  (K=151936, N=1536!)")
    wh = mk(V, H); lg = mk(512, V); rows = mk(256, H)
    for lab in (256, 128, 64):
        fl = 2 * lab * V * H
        ms, tf = timeit(lambda: torch.mm(lg[:lab], wh), fl, min_ms=300)
        row(f"dX as in repo, lab={lab} (tn? nn)", ms, tf, f"model tiles={math.ceil(H / 256) * math.ceil(lab / 128)} of {SM} SMs")
        ms2, tf2 = timeit(lambda: torch.mm(wh.t(), lg[:lab].t()), fl, min_ms=300)
        row(f"dX transposed formulation, lab={lab}", ms2, tf2)
        for S in (2, 4, 8):
            a = lg[:lab].view(lab, S, V // S).transpose(0, 1)       # [S, lab, V/S] strided view
            b = wh.view(S, V // S, H)
            ms3, tf3 = timeit(lambda: torch.bmm(a, b).sum(0, dtype=torch.float32), fl, min_ms=300)
            row(f"dX split-K bmm S={S} + fp32 sum, lab={lab}", ms3, tf3, f"vs repo {ms - ms3:+.3f} ms")
    ms, tf = timeit(lambda: torch.mm(lg[:512], wh), 2 * 512 * V * H, min_ms=300)
    row("dX with 512 rows at once (buffer +74 MiB)", ms, tf, f"per 256 rows: {ms / 2:.3f} ms")
    out = torch.empty(256, V, device=dev, dtype=dt)
    ms, tf = timeit(lambda: torch.mm(rows, wh.t(), out=out), 2 * 256 * V * H, min_ms=300)
    row("logits fwd 256 rows (tn)", ms, tf)
    a = lg[:256].view(256, 4, V // 4).transpose(0, 1)
    ref = torch.mm(lg[:256], wh).float()
    alt = torch.bmm(a, wh.view(4, V // 4, H)).sum(0, dtype=torch.float32)
    print(f"  split-K S=4 vs repo mm: max abs diff {(ref - alt).abs().max().item():.4g} (output scale {ref.abs().max().item():.4g})")
    print("  gpu:", gpu_state())


# ---------------------------------------------------------------- chain: the whole base GEMM chain of one step
def sec_chain(Ms):
    print("\n== chain: 28 layers fwd (F.linear) then 28 layers dX (matmul) with distinct weights, + head chunks; compare with ~265-291 ms measured in the step")
    L = []
    for _ in range(28):
        L.append(dict(q=mk(H, H), k=mk(KV, H), v=mk(KV, H), o=mk(H, H), g=mk(I, H), u=mk(I, H), d=mk(H, I),
                      bq=mk(H), bk=mk(KV), bv=mk(KV)))
    wh = mk(V, H); lg = mk(256, V)
    for M in Ms:
        x, hI, gH, gK, gI = mk(M, H), mk(M, I), mk(M, H), mk(M, KV), mk(M, I)
        lab = int(0.6 * M)

        def fwd():
            for l in L:
                F.linear(x, l["q"], l["bq"]); F.linear(x, l["k"], l["bk"]); F.linear(x, l["v"], l["bv"]); F.linear(x, l["o"])
                F.linear(x, l["g"]); F.linear(x, l["u"]); F.linear(hI, l["d"])

        def head():
            for s in range(0, M, 256):
                n = min(256, M - s)
                torch.mm(x[:n], wh.t(), out=lg[:n])
            for s in range(0, lab, 256):
                torch.mm(lg[:min(256, lab - s)], wh)

        def bwd():  # dX of every linear: grad[M,out] @ W[out,in]
            for l in reversed(L):
                torch.matmul(gH, l["q"]); torch.matmul(gK, l["k"]); torch.matmul(gK, l["v"]); torch.matmul(gH, l["o"])
                torch.matmul(gI, l["g"]); torch.matmul(gI, l["u"]); torch.matmul(gH, l["d"])
        macs = 28 * (2 * H * H + 2 * H * KV + 3 * H * I)
        fl_f, fl_b = 2 * M * macs, 2 * M * macs
        fl_h = 2 * M * V * H + 2 * lab * V * H
        a, tfa = timeit(fwd, fl_f, min_ms=400, reps=2)
        b, tfb = timeit(bwd, fl_b, min_ms=400, reps=2)
        h, tfh = timeit(head, fl_h, min_ms=400, reps=2)
        tot = a + b + h
        print(f"  M={M}: fwd {a:.1f} ms ({tfa:.1f} TF)  dX {b:.1f} ms ({tfb:.1f} TF)  head {h:.1f} ms ({tfh:.1f} TF)  total {tot:.1f} ms = {(fl_f + fl_b + fl_h) / tot / 1e9:.1f} TFLOP/s")
    print("  gpu:", gpu_state())


# ---------------------------------------------------------------- layout: tn vs nn for the same maths
def sec_layout(Ms):
    print("\n== layout: same maths with the weight stored as [out,in] (repo) or transposed [in,out] (cuBLAS flips tn <-> nn)")
    for M in Ms:
        print(f" M={M}")
        for name, N, K in [("gate fwd", I, H), ("o fwd", H, H), ("down fwd", H, I)]:
            w = rot(lambda: mk(N, K)); wt = rot(lambda: mk(K, N)); x = mk(M, K)
            a, tfa = timeit(lambda: F.linear(x, w()), 2 * M * N * K, min_ms=120)
            b, tfb = timeit(lambda: x @ wt(), 2 * M * N * K, min_ms=120)
            print(f"  {name:<9s} fwd: x@W.T ('tn', repo) {a:.3f} ms {tfa:.1f} TF | x@Wt ('nn') {b:.3f} ms {tfb:.1f} TF")
        for name, N, K in [("gate dX", H, I), ("o dX", H, H), ("down dX", I, H)]:
            w = rot(lambda: mk(K, N)); wt = rot(lambda: mk(N, K)); g = mk(M, K)
            a, tfa = timeit(lambda: g @ w(), 2 * M * N * K, min_ms=120)
            b, tfb = timeit(lambda: g @ wt().t(), 2 * M * N * K, min_ms=120)
            print(f"  {name:<9s}: g@W ('nn', repo) {a:.3f} ms {tfa:.1f} TF | g@Wt.T ('tn') {b:.3f} ms {tfb:.1f} TF")
    print("  gpu:", gpu_state())


SECTIONS = dict(layer=sec_layer, pad=sec_pad, sweep=sec_sweep, layout=sec_layout, concat=sec_concat, streams=sec_streams,
                knobs=sec_knobs, head=sec_head, chain=sec_chain)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="chain,head,layer,pad,sweep,layout,concat,streams,knobs")
    ap.add_argument("--Ms", default="826,1170,1500,2048")
    args = ap.parse_args()
    Ms = [int(m) for m in args.Ms.split(",")]
    print(torch.__version__, torch.cuda.get_device_name(0), "cuda", torch.version.cuda)
    print("gpu at start:", gpu_state())
    # warm the GPU so the clock settles near its sustained value before the first number
    a, b = mk(2048, 2048), mk(2048, 2048)
    for _ in range(300):
        a @ b
    torch.cuda.synchronize()
    print("gpu after warm-up:", gpu_state())
    for s in args.sections.split(","):
        SECTIONS[s](Ms)
