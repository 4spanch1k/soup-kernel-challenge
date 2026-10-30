"""Qwen2 layer-tail fusion against the unfused PyTorch operations."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def reference(residual, delta, weight, eps, y_dtype):
    total = residual + delta
    x = total.float()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * normed.to(total.dtype)).to(y_dtype), total


@pytest.mark.parametrize("y_dtype", [torch.float16, torch.float32])
def test_fused_boundary_matches_pytorch_forward_and_backward(y_dtype):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    from solution.layer_tail import _fuse_add_norm

    torch.manual_seed(0)
    shape = (2, 5, 1001)
    residual = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    delta = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.rand(shape[-1], device="cuda") + 0.5
    norm = Qwen2RMSNorm(shape[-1]).cuda()
    with torch.no_grad():
        norm.weight.copy_(weight)
    dy = torch.randn(shape, device="cuda", dtype=y_dtype)
    dsum = torch.randn(shape, device="cuda", dtype=torch.float32)

    ref_residual = residual.detach().clone().requires_grad_(True)
    ref_delta = delta.detach().clone().requires_grad_(True)
    ref_y, ref_total = reference(ref_residual, ref_delta, weight, norm.variance_epsilon, y_dtype)
    torch.autograd.backward((ref_y, ref_total), (dy, dsum))

    with torch.autocast("cuda", dtype=torch.float16, enabled=y_dtype == torch.float16):
        y, total = _fuse_add_norm(residual, delta, norm)
    torch.autograd.backward((y, total), (dy, dsum))

    assert y.dtype == y_dtype and total.dtype == torch.float32
    torch.testing.assert_close(y, ref_y, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(total, ref_total, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(residual.grad, ref_residual.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(delta.grad.float(), ref_delta.grad.float(), rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("num_layers", [3, 1])
def test_model_forward_matches_original(num_layers):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.models.qwen2 import modeling_qwen2

    from solution.add_rmsnorm import patch_qwen2_decoder_layer
    from solution.layer_tail import patch_qwen2_layer_tail
    from solution.rmsnorm import patch_qwen2_rmsnorm

    config = Qwen2Config(vocab_size=503, hidden_size=256, intermediate_size=512,
                         num_hidden_layers=num_layers, num_attention_heads=2,
                         num_key_value_heads=1, use_cache=False)
    torch.manual_seed(0)
    reference_model = Qwen2ForCausalLM(config).cuda().train()
    fused_model = Qwen2ForCausalLM(config).cuda().train()
    fused_model.load_state_dict(reference_model.state_dict())
    input_ids = torch.randint(0, config.vocab_size, (2, 17), device="cuda")
    current_forward = modeling_qwen2.Qwen2Model.forward
    original_forward = getattr(current_forward, "_soup_layer_tail_original", current_forward)

    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_layer()
    patch_qwen2_layer_tail()
    fused_forward = modeling_qwen2.Qwen2Model.forward

    def run(model):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        loss.backward()
        layer_index = 1 if num_layers == 3 else 0
        grad = model.model.layers[layer_index].self_attn.q_proj.weight.grad.detach().clone()
        return loss.detach(), grad

    try:
        modeling_qwen2.Qwen2Model.forward = original_forward
        base_loss, base_grad = run(reference_model)
    finally:
        modeling_qwen2.Qwen2Model.forward = fused_forward

    fused_loss, fused_grad = run(fused_model)
    torch.testing.assert_close(fused_loss, base_loss, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(fused_grad, base_grad, rtol=5e-2, atol=5e-2)
