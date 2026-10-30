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
