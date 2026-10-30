"""Fused NF4 matrix multiplies against dequantization followed by PyTorch GEMM."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

SHAPES = [(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("m", [1, 17, 826])
def test_forward_and_input_gradient_match_reference(shape, m):
    import bitsandbytes.functional as bnb_functional

    from solution.fused4bit import fused_matmul_forward, fused_matmul_input_grad
    from solution.nf4_dequant import nf4_dequantize

    n, k = shape
    torch.manual_seed(0)
    weight = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.02
    packed, state = bnb_functional.quantize_4bit(
        weight, blocksize=64, compress_statistics=True, quant_type="nf4")
    dequantized = nf4_dequantize(packed, state)

    x = torch.randn((m, k), device="cuda", dtype=torch.float16)
    dy = torch.randn((m, n), device="cuda", dtype=torch.float16)
    expected_y = torch.nn.functional.linear(x, dequantized)
    expected_dx = torch.matmul(dy, dequantized)

    y = fused_matmul_forward(x, packed, state)
    dx = fused_matmul_input_grad(dy, packed, state)

    torch.testing.assert_close(y, expected_y, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dx, expected_dx, rtol=5e-3, atol=5e-3)


def test_matmul4bit_patch_matches_forward_input_and_bias_gradients():
    import bitsandbytes.functional as bnb_functional
    from bitsandbytes.autograd import _functions

    from solution import fused4bit

    torch.manual_seed(0)
    n, k, m = 256, 1536, 17
    weight = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.02
    packed, state = bnb_functional.quantize_4bit(
        weight, blocksize=64, compress_statistics=True, quant_type="nf4")
    x_data = torch.randn((m, k), device="cuda", dtype=torch.float16)
    bias_data = torch.randn((n,), device="cuda", dtype=torch.float16)

    def run():
        x = x_data.clone().requires_grad_(True)
        bias = bias_data.clone().requires_grad_(True)
        y = bnb_functional.matmul_4bit(x, packed.t(), quant_state=state, bias=bias)
        y.float().square().mean().backward()
        return y.detach(), x.grad.detach(), bias.grad.detach()

    reference = run()
    original_forward = vars(_functions.MatMul4Bit)["forward"]
    original_backward = vars(_functions.MatMul4Bit)["backward"]
    fused4bit.ENABLED = True
    try:
        fused4bit.patch_fused4bit()
        actual = run()
    finally:
        _functions.MatMul4Bit.forward = original_forward
        _functions.MatMul4Bit.backward = original_backward
        fused4bit.ENABLED = False

    for got, expected in zip(actual, reference):
        torch.testing.assert_close(got, expected, rtol=5e-3, atol=5e-3)
