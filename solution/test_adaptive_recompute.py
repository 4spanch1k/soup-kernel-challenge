"""Adaptive MLP recompute: same loss and gradients as the plain MLP, and it only fires above the threshold."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("rows,threshold,expect_checkpoint", [(40, 30, True), (20, 30, False)])
def test_adaptive_recompute_matches_plain_mlp(monkeypatch, rows, threshold, expect_checkpoint):
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2

    import solution.recompute as recompute

    config = Qwen2Config(hidden_size=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=1)
    torch.manual_seed(0)
    original = modeling_qwen2.Qwen2MLP.forward
    mlp = modeling_qwen2.Qwen2MLP(config).cuda().train()
    x = torch.randn(rows, 64, device="cuda", requires_grad=True)

    ref = mlp(x).float().pow(2).sum()
    ref_grad, = torch.autograd.grad(ref, x)

    monkeypatch.setenv("SOUP_X_ADAPT_RECOMP", str(threshold))
    calls = []
    real_checkpoint = recompute.checkpoint
    monkeypatch.setattr(recompute, "checkpoint", lambda *a, **k: (calls.append(1), real_checkpoint(*a, **k))[1])
    try:
        recompute.patch_qwen2_mlp_adaptive_recompute()
        out = mlp(x).float().pow(2).sum()
        grad, = torch.autograd.grad(out, x)
    finally:
        modeling_qwen2.Qwen2MLP.forward = original
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grad, ref_grad, rtol=1e-4, atol=1e-4)
    assert bool(calls) is expect_checkpoint
