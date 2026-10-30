"""The recomputed MLP gives the same output and gradients, and keeps less for backward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_recompute_matches_and_saves_memory():
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2

    from solution.recompute import patch_qwen2_mlp_recompute

    config = Qwen2Config(hidden_size=256, intermediate_size=2048)
    torch.manual_seed(0)
    mlp = modeling_qwen2.Qwen2MLP(config).cuda().train()
    x = torch.randn(4, 256, 256, device="cuda", requires_grad=True)

    def run():
        mlp.zero_grad()
        x.grad = None
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.float16):
            out = mlp(x)
        kept = torch.cuda.memory_allocated() - before
        out.float().square().sum().backward()
        return out.detach().clone(), x.grad.clone(), mlp.up_proj.weight.grad.clone(), kept

    out0, gx0, gw0, kept0 = run()
    patch_qwen2_mlp_recompute(force=True)
    out1, gx1, gw1, kept1 = run()
    torch.testing.assert_close(out1, out0, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(gx1, gx0, rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(gw1, gw0, rtol=2e-2, atol=1e-2)
    assert kept1 < kept0
