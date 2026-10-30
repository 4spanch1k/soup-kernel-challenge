"""The dtype autocast would use for the matmuls of this call, or None when autocast is off."""

import torch


def autocast_dtype():
    try:
        if torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
    except (TypeError, AttributeError):  # older torch: no device argument
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return None
