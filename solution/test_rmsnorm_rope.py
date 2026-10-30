"""RMSNorm and RoPE kernels against the PyTorch code of transformers' Qwen2, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def reference_rmsnorm(x, weight, eps):
    """Qwen2RMSNorm.forward, copied."""
    dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(dtype)


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1001)])
def test_rmsnorm_matches_pytorch(x_dtype, shape):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda") * 3).to(x_dtype).requires_grad_(True)
    weight = (torch.rand(shape[-1], device="cuda") + 0.5).requires_grad_(True)  # fp32 like Soup
    grad = torch.randn(shape, device="cuda", dtype=torch.promote_types(x_dtype, torch.float32))

    ref_x, ref_w = x.detach().clone().requires_grad_(True), weight.detach().clone().requires_grad_(True)
    ref = reference_rmsnorm(ref_x, ref_w, 1e-6)
    ref.backward(grad)

    out = rmsnorm(x, weight, 1e-6)
    out.backward(grad)

    assert out.dtype == ref.dtype
    tol = 2e-3 if x_dtype == torch.float16 else 1e-5
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_x.grad.float(), rtol=5 * tol, atol=5 * tol)
    torch.testing.assert_close(weight.grad, ref_w.grad, rtol=1e-3, atol=1e-2)


def test_rmsnorm_frozen_weight_gets_no_grad():
    from solution.rmsnorm import rmsnorm

    x = torch.randn(2, 8, 64, device="cuda", requires_grad=True)
    weight = torch.ones(64, device="cuda")
    rmsnorm(x, weight, 1e-6).sum().backward()
    assert weight.grad is None and x.grad is not None


def reference_rope(q, k, cos, sin):
    """apply_rotary_pos_emb, copied."""
    def rotate_half(t):
        return torch.cat((-t[..., t.shape[-1] // 2:], t[..., : t.shape[-1] // 2]), dim=-1)

    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


@pytest.mark.parametrize("q_dtype", [torch.float16, torch.float32])
def test_rope_matches_pytorch(q_dtype):
    from solution.rope import RotaryFunction

    torch.manual_seed(0)
    batch, seq, heads, kv_heads, dim = 3, 61, 12, 2, 128
    # q and k are views of a [B, T, heads * D] projection, as in Qwen2Attention.
    q_proj = torch.randn(batch, seq, heads * dim, device="cuda", dtype=q_dtype)
    k_proj = torch.randn(batch, seq, kv_heads * dim, device="cuda", dtype=q_dtype)
    q = q_proj.view(batch, seq, heads, dim).transpose(1, 2).requires_grad_(True)
    k = k_proj.view(batch, seq, kv_heads, dim).transpose(1, 2).requires_grad_(True)
    freqs = torch.randn(batch, seq, dim // 2, device="cuda")
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()  # fp32, like the model under autocast

    ref_q, ref_k = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True)
    rq, rk = reference_rope(ref_q, ref_k, cos, sin)
    gq, gk = torch.randn_like(rq), torch.randn_like(rk)
    (rq * gq).sum().backward(retain_graph=False)
    (rk * gk).sum().backward()

    oq = RotaryFunction.apply(q, cos, sin, rq.dtype)
    ok = RotaryFunction.apply(k, cos, sin, rk.dtype)
    (oq * gq).sum().backward()
    (ok * gk).sum().backward()

    tol = 2e-3 if q_dtype == torch.float16 else 1e-5
    torch.testing.assert_close(oq, rq, rtol=tol, atol=tol)
    torch.testing.assert_close(ok, rk, rtol=tol, atol=tol)
    torch.testing.assert_close(q.grad.float(), ref_q.grad.float(), rtol=5 * tol, atol=5 * tol)
    torch.testing.assert_close(k.grad.float(), ref_k.grad.float(), rtol=5 * tol, atol=5 * tol)


def test_patches_match_a_tiny_qwen2():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution.rmsnorm import patch_qwen2_rmsnorm
    from solution.rope import patch_qwen2_rope

    config = Qwen2Config(vocab_size=503, hidden_size=256, intermediate_size=512,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).cuda().train()
    ids = torch.randint(0, 503, (2, 33), device="cuda")

    def run():
        model.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        return loss.detach(), model.model.layers[0].self_attn.q_proj.weight.grad.clone()

    base_loss, base_grad = run()
    patch_qwen2_rmsnorm()
    patch_qwen2_rope()
    loss, grad = run()
    torch.testing.assert_close(loss, base_loss, rtol=2e-3, atol=2e-3)
    assert (grad - base_grad).abs().max() <= 5e-2 * base_grad.abs().max()
