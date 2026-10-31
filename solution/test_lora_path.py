"""The optional PEFT 4-bit LoRA update against its original forward."""

import copy
import sys
import types

import pytest
import torch
from torch import nn


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


class FakeBaseLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x).to(x.dtype)


class FakeLinear4bit(nn.Module):
    """Small PEFT-shaped layer so this test does not need a quantized model download."""

    def __init__(self, width, output, rank):
        super().__init__()
        self.base_layer = FakeBaseLinear(width, output, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(width, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, output, bias=False)})
        self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(0.05)})
        self.scaling = {"default": 0.375}
        self.lora_variant = {}
        self.lora_bias = {"default": False}
        self.use_dora = {"default": False}
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged = False
        self.base_layer.weight.requires_grad_(False)

    def forward(self, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        if self.disable_adapters or self.merged:
            return result
        result = result.clone()
        for adapter in self.active_adapters:
            if adapter not in self.lora_A:
                continue
            lora_a = self.lora_A[adapter]
            lora_b = self.lora_B[adapter]
            dropout = self.lora_dropout[adapter]
            scaling = self.scaling[adapter]
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(lora_a.weight.dtype)
            output = lora_b(lora_a(dropout(x))) * scaling
            if requires_conversion:
                output = output.to(expected_dtype)
            result = result + output
        return result


ORIGINAL_FORWARD = FakeLinear4bit.forward


def _install_fake_peft(monkeypatch):
    peft = types.ModuleType("peft")
    bnb_lora = types.ModuleType("peft.tuners.lora.bnb")
    bnb_lora.Linear4bit = FakeLinear4bit
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "peft.tuners.lora.bnb", bnb_lora)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(7, 13), (2, 5, 13)])
def test_forward_and_backward_match_original(monkeypatch, dtype, shape):
    from solution.lora_path import patch_lora_linear4bit

    monkeypatch.setenv("SOUP_X_LORA", "1")
    FakeLinear4bit.forward = ORIGINAL_FORWARD
    _install_fake_peft(monkeypatch)
    reference = FakeLinear4bit(13, 11, 5).cuda().train()
    patched = copy.deepcopy(reference)
    base_outputs = []
    patched.base_layer.register_forward_hook(
        lambda _module, _inputs, output: base_outputs.append(output))
    torch.manual_seed(23)
    x_ref = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
    x_fast = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(*shape[:-1], 11, device="cuda", dtype=torch.float16)

    with torch.autocast("cuda", dtype=torch.float16):
        torch.cuda.manual_seed(91)
        y_ref = reference(x_ref)
        (y_ref * grad).sum().backward()

        patch_lora_linear4bit()
        torch.cuda.manual_seed(91)
        y_fast = patched(x_fast)
        (y_fast * grad).sum().backward()

    assert y_fast.data_ptr() != base_outputs[-1].data_ptr()
    tolerance = 1e-2 if dtype == torch.float16 else 2e-3
    torch.testing.assert_close(y_fast, y_ref, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(x_fast.grad, x_ref.grad, rtol=2 * tolerance, atol=2 * tolerance)
    for name, parameter in patched.named_parameters():
        if name.startswith("lora_"):
            expected = dict(reference.named_parameters())[name].grad
            torch.testing.assert_close(parameter.grad, expected, rtol=2 * tolerance,
                                       atol=2 * tolerance)


@pytest.mark.parametrize("unsupported", ["autocast_off", "disabled", "merged", "multiple",
                                           "dora", "bias"])
def test_unsupported_modes_match_original(monkeypatch, unsupported):
    from solution.lora_path import patch_lora_linear4bit

    monkeypatch.setenv("SOUP_X_LORA", "1")
    FakeLinear4bit.forward = ORIGINAL_FORWARD
    _install_fake_peft(monkeypatch)
    reference = FakeLinear4bit(13, 11, 5).cuda().train()
    patched = copy.deepcopy(reference)
    if unsupported == "disabled":
        reference.disable_adapters = patched.disable_adapters = True
    elif unsupported == "merged":
        reference.merged = patched.merged = True
    elif unsupported == "multiple":
        for layer in (reference, patched):
            layer.lora_A["other"] = copy.deepcopy(layer.lora_A["default"])
            layer.lora_B["other"] = copy.deepcopy(layer.lora_B["default"])
            layer.lora_dropout["other"] = copy.deepcopy(layer.lora_dropout["default"])
            layer.scaling["other"] = layer.scaling["default"]
            layer.lora_bias["other"] = False
            layer.use_dora["other"] = False
            layer.active_adapters = ["default", "other"]
    elif unsupported == "dora":
        reference.use_dora["default"] = patched.use_dora["default"] = True
    elif unsupported == "bias":
        reference.lora_bias["default"] = patched.lora_bias["default"] = True

    x_ref = torch.randn(5, 13, device="cuda", requires_grad=True)
    x_fast = x_ref.detach().clone().requires_grad_(True)
    enabled = unsupported != "autocast_off"
    grad = torch.randn(5, 11, device="cuda",
                       dtype=torch.float16 if enabled else torch.float32)
    with torch.autocast("cuda", dtype=torch.float16, enabled=enabled):
        torch.cuda.manual_seed(17)
        y_ref = reference(x_ref)
        y_ref.backward(grad)
        patch_lora_linear4bit()
        torch.cuda.manual_seed(17)
        y_fast = patched(x_fast)
        y_fast.backward(grad)

    torch.testing.assert_close(y_fast, y_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x_fast.grad, x_ref.grad, rtol=1e-5, atol=1e-5)


def test_fake_peft_module_without_bnb_class_is_ignored(monkeypatch):
    from solution.lora_path import patch_lora_linear4bit

    monkeypatch.setenv("SOUP_X_LORA", "1")
    monkeypatch.setitem(sys.modules, "peft", types.ModuleType("peft"))
    monkeypatch.delitem(sys.modules, "peft.tuners.lora.bnb", raising=False)
    patch_lora_linear4bit()
