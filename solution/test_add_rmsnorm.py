"""Fused residual add + RMSNorm against the PyTorch ops it replaces, forward and backward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def reference(residual, delta, weight, eps, y_dtype):
    total = residual + delta
    x = total.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * x.to(total.dtype)).to(y_dtype), total


@pytest.mark.parametrize("y_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1001)])
def test_matches_pytorch(y_dtype, shape):
    from solution.add_rmsnorm import add_rmsnorm

    torch.manual_seed(0)
    residual = (torch.randn(shape, device="cuda") * 2).requires_grad_(True)  # fp32 stream
    delta = torch.randn(shape, device="cuda", dtype=torch.float16).requires_grad_(True)
    weight = torch.rand(shape[-1], device="cuda") + 0.5
    dy = torch.randn(shape, device="cuda", dtype=y_dtype)
    dsum = torch.randn(shape, device="cuda")

    r2, d2 = residual.detach().clone().requires_grad_(True), delta.detach().clone().requires_grad_(True)
    ref_y, ref_total = reference(r2, d2, weight, 1e-6, y_dtype)
    torch.autograd.backward([ref_y, ref_total], [dy, dsum])

    y, total = add_rmsnorm(residual, delta, weight, 1e-6, y_dtype)
    torch.autograd.backward([y, total], [dy, dsum])

    assert y.dtype == y_dtype and total.dtype == torch.float32
    tol = 2e-3 if y_dtype == torch.float16 else 1e-5
    torch.testing.assert_close(total, ref_total, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(y, ref_y, rtol=tol, atol=tol)
    assert residual.grad.dtype == torch.float32 and delta.grad.dtype == torch.float16
    torch.testing.assert_close(residual.grad, r2.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(delta.grad.float(), d2.grad.float(), rtol=5e-3, atol=5e-3)


def test_decoder_layer_patch_matches_tiny_qwen2():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution.add_rmsnorm import patch_qwen2_decoder_layer

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
    patch_qwen2_decoder_layer()
    loss, grad = run()
    torch.testing.assert_close(loss, base_loss, rtol=2e-3, atol=2e-3)
    assert (grad - base_grad).abs().max() <= 5e-2 * base_grad.abs().max()
