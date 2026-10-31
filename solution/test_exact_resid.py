"""Exact fp32 residual reconstruction against the normal saved-sum path."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("residual_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("y_dtype", [torch.float16, torch.float32])
def test_exact_residual_matches_saved_sum(residual_dtype, y_dtype, monkeypatch):
    # Bit-equality compares against the unrounded saved sum, so the default fp16 saved-norm copy is off here.
    monkeypatch.setenv("SOUP_X_NORM_SAVE16", "0")
    from solution.add_rmsnorm import add_rmsnorm

    torch.manual_seed(11)
    shape = (3, 7, 1001)
    residual = torch.randn(shape, device="cuda", dtype=residual_dtype)
    delta = torch.randn(shape, device="cuda", dtype=torch.float16)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32)
    dy = torch.randn(shape, device="cuda", dtype=y_dtype)
    dsum = torch.randn(shape, device="cuda", dtype=torch.promote_types(residual_dtype, delta.dtype))

    def run(exact_residual):
        x = residual.detach().clone().requires_grad_(True)
        d = delta.detach().clone().requires_grad_(True)
        y, total = add_rmsnorm(x, d, weight, 1e-6, y_dtype, exact_residual)
        torch.autograd.backward((y, total), (dy, dsum))
        return y.detach(), total.detach(), x.grad, d.grad

    normal = run(False)
    exact = run(True)
    for normal_value, exact_value in zip(normal, exact):
        assert torch.equal(exact_value, normal_value)

    ref_residual = residual.detach().clone().requires_grad_(True)
    ref_delta = delta.detach().clone().requires_grad_(True)
    ref_total = ref_residual + ref_delta
    ref_x = ref_total.float()
    ref_x = ref_x * torch.rsqrt(ref_x.square().mean(-1, keepdim=True) + 1e-6)
    ref_y = (weight * ref_x.to(ref_total.dtype)).to(y_dtype)
    torch.autograd.backward((ref_y, ref_total), (dy, dsum))
    tolerance = 2e-3 if y_dtype == torch.float16 else 1e-5
    torch.testing.assert_close(normal[0], ref_y, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(normal[1], ref_total, rtol=0, atol=0)
    torch.testing.assert_close(normal[2], ref_residual.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(normal[3], ref_delta.grad, rtol=5e-3, atol=5e-3)
