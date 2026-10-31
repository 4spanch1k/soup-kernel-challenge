"""Checkpoint recomputation keeps dropout masks and gradients aligned."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scope", ["full", "sdpa"])
def test_checkpoint_attention_matches_direct_path(dtype, scope):
    from solution.layer_tail import _call_with_checkpoint

    def run(enabled):
        torch.manual_seed(31)
        x = torch.randn(2, 7, 16, device="cuda", dtype=dtype, requires_grad=True)
        qkv_weight = torch.randn(48, 16, device="cuda", dtype=dtype, requires_grad=True)
        out_weight = torch.randn(16, 16, device="cuda", dtype=dtype, requires_grad=True)

        def attention(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.2)

        def full_attention(hidden):
            q, k, v = F.linear(hidden, qkv_weight).chunk(3, dim=-1)
            q = F.dropout(q, p=0.2, training=True).view(2, 7, 2, 8).transpose(1, 2)
            k = k.view(2, 7, 2, 8).transpose(1, 2)
            v = v.view(2, 7, 2, 8).transpose(1, 2)
            use_inner_checkpoint = enabled and scope == "sdpa"
            attended = _call_with_checkpoint(
                attention, q, k, v, enabled=use_inner_checkpoint)
            merged = attended.transpose(1, 2).reshape(2, 7, 16)
            return F.linear(merged, out_weight)

        torch.cuda.manual_seed_all(71)
        with torch.autocast("cuda", dtype=torch.float16):
            if scope == "full":
                output = _call_with_checkpoint(full_attention, x, enabled=enabled)
            else:
                output = full_attention(x)
            loss = output.float().square().mean()
        loss.backward()
        return (output.detach(), x.grad, qkv_weight.grad, out_weight.grad,
                torch.cuda.get_rng_state())

    direct = run(False)
    checkpointed = run(True)
    for direct_value, checkpoint_value in zip(direct, checkpointed):
        assert torch.equal(checkpoint_value, direct_value)
