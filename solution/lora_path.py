"""Optional in-place LoRA accumulation for PEFT's bitsandbytes 4-bit layer."""

import importlib
import os

import torch


def _supported(self, cls, x, args, kwargs):
    if (type(self) is not cls or not isinstance(self, torch.nn.Module)
            or not torch.is_tensor(x) or args or kwargs
            or not torch.is_autocast_enabled() or not x.is_cuda or x.dim() not in (2, 3)
            or x.dtype not in (torch.float16, torch.float32)
            or getattr(self, "disable_adapters", True) or getattr(self, "merged", True)):
        return None

    adapters = getattr(self, "active_adapters", ())
    if not isinstance(adapters, (list, tuple)) or len(adapters) != 1:
        return None
    adapter = adapters[0]

    lora_a_layers = getattr(self, "lora_A", None)
    lora_b_layers = getattr(self, "lora_B", None)
    dropout_layers = getattr(self, "lora_dropout", None)
    scaling_values = getattr(self, "scaling", None)
    variants = getattr(self, "lora_variant", {})
    biases = getattr(self, "lora_bias", {})
    dora = getattr(self, "use_dora", {})
    if (lora_a_layers is None or lora_b_layers is None or dropout_layers is None
            or scaling_values is None or adapter not in scaling_values
            or adapter not in lora_a_layers or adapter not in lora_b_layers
            or adapter not in dropout_layers or variants is None or adapter in variants
            or not hasattr(biases, "get") or not hasattr(dora, "get")
            or bool(biases.get(adapter, False)) or bool(dora.get(adapter, False))):
        return None
    lora_a = lora_a_layers[adapter]
    lora_b = lora_b_layers[adapter]
    dropout = dropout_layers[adapter]
    scaling = scaling_values[adapter]
    if (not isinstance(lora_a, torch.nn.Linear) or not isinstance(lora_b, torch.nn.Linear)
            or not isinstance(dropout, torch.nn.Module)
            or lora_a.bias is not None or lora_b.bias is not None
            or not isinstance(scaling, (float, int))
            or lora_a.weight.device != x.device or lora_b.weight.device != x.device):
        return None
    return lora_a, lora_b, dropout, scaling


def patch_lora_linear4bit():
    """Patch only the enabled, supported PEFT Linear4bit path; keep its forward as fallback."""
    if os.environ.get("SOUP_X_LORA") != "1":
        return

    try:
        importlib.import_module("peft")
        bnb_lora = importlib.import_module("peft.tuners.lora.bnb")
    except Exception:
        return

    cls = getattr(bnb_lora, "Linear4bit", None)
    if not isinstance(cls, type) or not issubclass(cls, torch.nn.Module):
        return
    if getattr(cls.forward, "_soup_x_lora_patched", False):
        return

    original = cls.forward

    def forward(self, x, *args, **kwargs):
        state = _supported(self, cls, x, args, kwargs)
        if state is None:
            return original(self, x, *args, **kwargs)

        lora_a, lora_b, dropout, scaling = state
        result = self.base_layer(x, *args, **kwargs)
        if not (torch.is_tensor(result) and result.is_contiguous()
                and result.shape[:-1] == x.shape[:-1]
                and result.shape[-1] == lora_b.out_features):
            return original(self, x, *args, **kwargs)
        if torch.promote_types(result.dtype, torch.get_autocast_dtype("cuda")) != result.dtype:
            return original(self, x, *args, **kwargs)

        update = lora_b(lora_a(dropout(x)))
        return result.add_(update, alpha=scaling)

    forward._soup_x_lora_patched = True
    forward._soup_x_lora_original = original
    cls.forward = forward
