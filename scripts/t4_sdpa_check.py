"""Which scaled_dot_product_attention backends work on this GPU for the shapes of our run, and how long
do they take (forward + backward)? Shapes: batch 4, 12 query heads, 2 KV heads, head_dim 128, T=512."""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

B, HQ, HKV, T, D = 4, 12, 2, 512, 128


def bench(fn, iters=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def make(requires_grad=True):
    q = torch.randn(B, HQ, T, D, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
    k = torch.randn(B, HKV, T, D, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
    v = torch.randn(B, HKV, T, D, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
    return q, k, v


def repeat_kv(x):
    return x[:, :, None].expand(B, HKV, HQ // HKV, T, D).reshape(B, HQ, T, D)


def case(name, backend, call):
    q, k, v = make()
    def step():
        out = call(q, k, v)
        out.sum().backward()
    try:
        with sdpa_kernel(backend):
            ms = bench(step)
            torch.cuda.reset_peak_memory_stats()
            step()
            mem = torch.cuda.max_memory_allocated() / 2**20
        print(f"{name:55s} {backend.name:20s} {ms:7.2f} ms fwd+bwd   peak {mem:7.1f} MiB")
    except Exception as exc:  # noqa: BLE001
        print(f"{name:55s} {backend.name:20s} NOT AVAILABLE: {str(exc).splitlines()[0][:90]}")


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    mask = torch.zeros(B, 1, T, T, device="cuda", dtype=torch.float16)
    mask.masked_fill_(torch.ones(T, T, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        case("repeat_kv + 4D mask", backend,
             lambda q, k, v: F.scaled_dot_product_attention(q, repeat_kv(k), repeat_kv(v), attn_mask=mask))
        case("repeat_kv + is_causal", backend,
             lambda q, k, v: F.scaled_dot_product_attention(q, repeat_kv(k), repeat_kv(v), is_causal=True))
        case("enable_gqa + is_causal (no repeat_kv)", backend,
             lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True))
    print("\ndefault dispatcher (what transformers gets without forcing a backend):")
    q, k, v = make()
    def default_mask():
        F.scaled_dot_product_attention(q, repeat_kv(k), repeat_kv(v), attn_mask=mask).sum().backward()
    def default_gqa():
        F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).sum().backward()
    for name, fn in (("repeat_kv + 4D mask", default_mask), ("enable_gqa + is_causal", default_gqa)):
        try:
            print(f"  {name:30s} {bench(fn):7.2f} ms fwd+bwd")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:30s} FAILED: {str(exc).splitlines()[0][:100]}")


if __name__ == "__main__":
    main()
