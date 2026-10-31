"""Exact fp32 residual reconstruction against the normal saved-sum path."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("residual_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("y_dtype", [torch.float16, torch.float32])
def test_exact_residual_matches_saved_sum(residual_dtype, y_dtype):
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
