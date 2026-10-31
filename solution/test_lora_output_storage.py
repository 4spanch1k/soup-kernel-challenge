"""Real bitsandbytes + PEFT coverage for the optional owning linear output."""

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _quantized_weight():
    import bitsandbytes.functional as bnb_functional

    torch.manual_seed(31)
    weight = (torch.randn(96, 128, device="cuda") * 0.02).half()
    return bnb_functional.quantize_4bit(
        weight, blocksize=64, compress_statistics=True, quant_type="nf4")


@pytest.mark.parametrize("shape", [(11, 128), (2, 7, 128)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_real_matmul_owning_output_and_inplace_backward(monkeypatch, shape, with_bias):
    import bitsandbytes as bnb

    from solution import nf4_dequant

    nf4_dequant.patch_bnb_dequantize()
    packed, state = _quantized_weight()
    x = torch.randn(*shape, device="cuda", dtype=torch.float16)
    bias = torch.randn(96, device="cuda", dtype=torch.float16) if with_bias else None
    grad = torch.randn(*shape[:-1], 96, device="cuda", dtype=torch.float16)

    def run(enabled):
        monkeypatch.setenv("SOUP_X_LINEAR_OUT", str(int(enabled)))
        a = x.detach().clone().requires_grad_(True)
        b = bias.detach().clone().requires_grad_(True) if bias is not None else None
        with torch.autocast("cuda", dtype=torch.float16):
            y = bnb.matmul_4bit(a, packed.t(), quant_state=state, bias=b)
            owning = y._base is None
            if enabled:
                # This is precisely what PEFT's clone removal needs to support.
                y.add_(torch.zeros_like(y))
            y.backward(grad)
        return y.detach(), a.grad.detach(), None if b is None else b.grad.detach(), owning

    expected, dx_expected, db_expected, _ = run(False)
    actual, dx, db, owning = run(True)
    assert owning
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(dx, dx_expected, rtol=5e-3, atol=5e-3)
    if with_bias:
        torch.testing.assert_close(db, db_expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(11, 128), (2, 7, 128)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_real_peft_lora_no_clone_matches_original(monkeypatch, dtype, shape, with_bias):
    import bitsandbytes as bnb
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit

    from solution.lora_path import patch_lora_linear4bit
    from solution.nf4_dequant import patch_bnb_dequantize

    original = getattr(Linear4bit.forward, "_soup_x_lora_original", Linear4bit.forward)
    monkeypatch.setattr(Linear4bit, "forward", original)
    patch_bnb_dequantize()

    torch.manual_seed(41)
    base = bnb.nn.Linear4bit(128, 96, bias=with_bias, compute_dtype=torch.float16,
                            compress_statistics=True, quant_type="nf4").half().cuda()
    base.requires_grad_(False)
    config = LoraConfig(r=16, lora_alpha=8, lora_dropout=0.05,
                        init_lora_weights=True, use_dora=False, bias="none")
    layer = Linear4bit(base, "default", config=config, r=config.r,
                       lora_alpha=config.lora_alpha).cuda().train()
    # PEFT initializes B to zero; nonzero weights exercise both adapter gradients.
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_(std=0.02)
        layer.lora_B["default"].weight.normal_(std=0.02)

    base_outputs = []
    handle = base.register_forward_hook(
        lambda _module, _inputs, output: base_outputs.append((output.data_ptr(), output._base)))
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    grad = torch.randn(*shape[:-1], 96, device="cuda", dtype=dtype)

    def run():
        layer.zero_grad(set_to_none=True)
        a = x.detach().clone().requires_grad_(True)
        torch.cuda.manual_seed(53)
        with torch.autocast("cuda", dtype=torch.float16):
            y = layer(a)
            y.backward(grad)
        grads = {name: p.grad.detach().clone() for name, p in layer.named_parameters()
                 if name.startswith("lora_") and p.requires_grad}
        return y.detach(), a.grad.detach(), grads

    monkeypatch.setenv("SOUP_X_LINEAR_OUT", "0")
    expected, dx_expected, grads_expected = run()
    monkeypatch.setenv("SOUP_X_LINEAR_OUT", "1")
    monkeypatch.setenv("SOUP_X_LORA", "1")
    patch_lora_linear4bit()
    actual, dx, grads = run()
    handle.remove()

    # Prove the real backend exercised the no-clone branch, including fp32 input
    # (Linear4bit casts to its fp16 compute dtype and casts the result back).
    assert actual.data_ptr() == base_outputs[-1][0]
    assert base_outputs[-1][1] is None
    tolerance = 1e-2 if dtype == torch.float16 else 2e-3
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(dx, dx_expected, rtol=2 * tolerance, atol=2 * tolerance)
    assert grads.keys() == grads_expected.keys() and len(grads) == 2
    for name, value in grads.items():
        torch.testing.assert_close(value, grads_expected[name], rtol=2 * tolerance,
                                   atol=2 * tolerance)


@pytest.mark.parametrize("unsupported", ["fp32", "noncontiguous"])
def test_real_matmul_unsupported_input_keeps_fallback(monkeypatch, unsupported):
    import bitsandbytes as bnb

    from solution import nf4_dequant

    nf4_dequant.patch_bnb_dequantize()
    packed, state = _quantized_weight()
    if unsupported == "fp32":
        x = torch.randn(2, 7, 128, device="cuda", dtype=torch.float32)
    else:
        x = torch.randn(7, 2, 128, device="cuda", dtype=torch.float16).transpose(0, 1)
        assert not x.is_contiguous()
    grad = torch.randn(2, 7, 96, device="cuda", dtype=x.dtype)

    def forbidden(*_args, **_kwargs):
        pytest.fail("unsupported input reached owning-output GEMM")

    monkeypatch.setattr(nf4_dequant, "_fresh_linear_output", forbidden)

    def run(enabled):
        monkeypatch.setenv("SOUP_X_LINEAR_OUT", str(int(enabled)))
        # preserve_format retains the noncontiguous stride in this test.
        a = x.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            y = bnb.matmul_4bit(a, packed.t(), quant_state=state)
            y.backward(grad)
        return y.detach(), a.grad.detach()

    expected, dx_expected = run(False)
    actual, dx = run(True)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(dx, dx_expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("mode", ["int8", "int4"])
@pytest.mark.parametrize("unsupported", ["empty", "float64"])
def test_saved_activation_codec_empty_and_dtype_guard(monkeypatch, mode, unsupported):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from solution import activation_codec

    calls = []

    def plain_forward(self, x):
        calls.append(True)
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    monkeypatch.setattr(Qwen2MLP, "forward", plain_forward)
    monkeypatch.setenv("SOUP_X_ACT_INT8", "1" if mode == "int8" else "0")
    monkeypatch.setenv("SOUP_X_ACT_INT4", "1" if mode == "int4" else "0")
    activation_codec.patch_saved_activation_int8()
    dtype = torch.float64 if unsupported == "float64" else torch.float16
    layer = Qwen2MLP(Qwen2Config(hidden_size=32, intermediate_size=64)).cuda().to(dtype)
    layer.train().requires_grad_(False)
    shape = (0, 32) if unsupported == "empty" else (3, 32)
    x = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)

    def forbidden(*_args):
        pytest.fail("unsupported activation entered the compression kernel")

    monkeypatch.setattr(activation_codec.SavedActivationInt8, "apply", forbidden)
    monkeypatch.setattr(activation_codec.SavedActivationInt4, "apply", forbidden)
    actual = layer(x)
    expected = plain_forward(layer, x)
    assert len(calls) == 2
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
