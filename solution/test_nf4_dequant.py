"""Our NF4 dequantization against bitsandbytes' own, on weights it quantized itself."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("shape", [(1536, 8960), (256, 1536), (1536, 1536), (192, 130)])
def test_matches_bitsandbytes(shape):
    import bitsandbytes.functional as bnb_functional

    from solution.nf4_dequant import nf4_dequantize

    torch.manual_seed(0)
    weight = (torch.randn(shape, device="cuda") * 0.02).half()
    packed, state = bnb_functional.quantize_4bit(weight, blocksize=64, compress_statistics=True,
                                                 quant_type="nf4")
    expected = bnb_functional.dequantize_4bit(packed, state)
    got = nf4_dequantize(packed, state)
    assert got.shape == expected.shape and got.dtype == expected.dtype
    diff = (got.float() - expected.float()).abs()
    # fp32 scale arithmetic can round differently by one fp32 ulp before the fp16 cast
    assert (diff == 0).float().mean() > 0.9999 and diff.max() <= 2 ** -10 * expected.abs().max()


def test_patch_is_used_and_falls_back():
    import bitsandbytes.functional as bnb_functional

    from solution import nf4_dequant

    weight = (torch.randn(128, 256, device="cuda") * 0.02).half()
    packed, state = bnb_functional.quantize_4bit(weight, blocksize=64, compress_statistics=True,
                                                 quant_type="nf4")
    reference = bnb_functional.dequantize_4bit(packed, state)
    nf4_dequant.patch_bnb_dequantize()
    patched = bnb_functional.dequantize_4bit(packed, state)
    torch.testing.assert_close(patched, reference, rtol=2e-3, atol=1e-5)
    plain, plain_state = bnb_functional.quantize_4bit(weight, blocksize=64, quant_type="nf4")
    torch.testing.assert_close(bnb_functional.dequantize_4bit(plain, plain_state),
                               weight, rtol=0.2, atol=0.05)  # not nested: bnb handles it


def test_matmul_4bit_forward_and_backward_match_bitsandbytes():
    """bnb.matmul_4bit (the call Linear4bit makes) before and after our patch: same output, same
    gradient with respect to the input, for a 3-D activation with a bias."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_functional

    from solution import nf4_dequant

    torch.manual_seed(0)
    weight = (torch.randn(256, 1536, device="cuda") * 0.02).half()
    packed, state = bnb_functional.quantize_4bit(weight, blocksize=64, compress_statistics=True,
                                                 quant_type="nf4")
    bias = torch.randn(256, device="cuda").half()
    x = torch.randn(3, 41, 1536, device="cuda").half()

    def run():
        a = x.clone().requires_grad_(True)
        out = bnb.matmul_4bit(a, packed.t(), quant_state=state, bias=bias)
        out.float().square().sum().backward()
        return out.detach(), a.grad.detach()

    ref_out, ref_grad = run()
    nf4_dequant.patch_bnb_dequantize()
    out, grad = run()
    torch.testing.assert_close(out, ref_out, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grad, ref_grad, rtol=5e-3, atol=5e-3)
