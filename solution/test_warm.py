import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_every_warm_variant_compiles_without_launching():
    from solution.warm import _variants

    variants = _variants()
    assert variants
    for kernel, args, kwargs in variants:
        kernel.warmup(*args, grid=(1,), **kwargs)
