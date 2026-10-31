"""Check the varlen attention path against padded causal SDPA."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from solution.layer_tail import _call_varlen_attention, _efficient_attention_operator


@pytest.mark.parametrize("lengths", [(3, 1, 5), (5, 5, 5), (7, 3, 1)])
def test_varlen_attention_matches_padded_sdpa(lengths):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for PyTorch efficient attention")
    operator = _efficient_attention_operator()
    if operator is None:
        pytest.skip("PyTorch efficient attention with autograd is unavailable")

    batch = len(lengths)
    max_length = max(lengths)
    heads, kv_heads, head_dim = 12, 2, 128
    scale = head_dim ** -0.5
    device = "cuda"
    valid = torch.arange(max_length, device=device)[None, :] < torch.tensor(
        lengths, device=device
    )[:, None]
    q_input = torch.randn(batch, max_length, heads, head_dim, device=device,
                          dtype=torch.float16)
    k_input = torch.randn(batch, max_length, kv_heads, head_dim, device=device,
                          dtype=torch.float16)
    v_input = torch.randn_like(k_input)
    q_var = q_input.detach().clone().requires_grad_()
    k_var = k_input.detach().clone().requires_grad_()
    v_var = v_input.detach().clone().requires_grad_()
    q_ref = q_input.detach().clone().requires_grad_()
    k_ref = k_input.detach().clone().requires_grad_()
    v_ref = v_input.detach().clone().requires_grad_()

    lengths_tensor = valid.sum(1, dtype=torch.int32)
    cu_seqlens = torch.cat((lengths_tensor.new_zeros(1), lengths_tensor.cumsum(0)))
    query = q_var[valid].unsqueeze(0)
    key = k_var[valid].unsqueeze(0).repeat_interleave(heads // kv_heads, dim=2)
    value = v_var[valid].unsqueeze(0).repeat_interleave(heads // kv_heads, dim=2)
    actual = _call_varlen_attention(operator, query, key, value, cu_seqlens,
                                    max_length, scale)
    assert actual is not None
    actual = actual.squeeze(0)

    q_ref_heads = q_ref.transpose(1, 2)
    k_ref_heads = k_ref.repeat_interleave(heads // kv_heads, dim=2).transpose(1, 2)
    v_ref_heads = v_ref.repeat_interleave(heads // kv_heads, dim=2).transpose(1, 2)
    causal = torch.arange(max_length, device=device)[:, None] >= torch.arange(
        max_length, device=device
    )[None, :]
    allowed = valid[:, None, None, :] & causal[None, None, :, :]
    reference = F.scaled_dot_product_attention(
        q_ref_heads, k_ref_heads, v_ref_heads, attn_mask=allowed,
        dropout_p=0.0, is_causal=False, scale=scale)
    reference = reference.transpose(1, 2)[valid]

    torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q_var, k_var, v_var), grad)
    reference_grads = torch.autograd.grad(reference, (q_ref, k_ref, v_ref), grad)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, atol=5e-2, rtol=5e-2)
