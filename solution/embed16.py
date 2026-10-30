"""Keep an exactly representable tied Qwen2 embedding and head in fp16."""

from types import MethodType

import torch
import torch.nn.functional as F

_CHECK_ELEMENTS = 1 << 24


def _same_storage(left, right):
    left_storage = left.untyped_storage()
    right_storage = right.untyped_storage()
    return (left.device == right.device and left.shape == right.shape
            and left.stride() == right.stride()
            and left.storage_offset() == right.storage_offset()
            and left_storage.data_ptr() == right_storage.data_ptr()
            and left_storage.nbytes() == right_storage.nbytes())


def _is_exact_fp16(weight):
    rows_per_chunk = max(1, _CHECK_ELEMENTS // weight.shape[1])
    for start in range(0, weight.shape[0], rows_per_chunk):
        chunk = weight[start:start + rows_per_chunk]
        finite = torch.isfinite(chunk).all()
        exact = torch.eq(chunk.half().float(), chunk).all()
        if not (finite & exact).item():
            return False
    return True


def _embedding_forward(self, input_ids):
    return F.embedding(input_ids, self.weight, self.padding_idx, self.max_norm,
                       self.norm_type, self.scale_grad_by_freq, self.sparse).float()


def compact_tied_embedding(model_core):
    """Compact a frozen, fp32, tied Qwen2 embedding/head when its values round-trip exactly."""
    if getattr(model_core, "_soup_embed16_checked", False):
        return False

    model = getattr(model_core, "model", None)
    embedding = getattr(model, "embed_tokens", None)
    lm_head = getattr(model_core, "lm_head", None)
    if not (isinstance(embedding, torch.nn.Embedding)
            and isinstance(lm_head, torch.nn.Linear) and lm_head.bias is None):
        return False

    embedding_weight, head_weight = embedding.weight, lm_head.weight
    if not (embedding_weight.ndim == 2 and embedding_weight.is_cuda
            and head_weight.is_cuda and embedding_weight.device == head_weight.device
            and embedding_weight.dtype == torch.float32
            and head_weight.dtype == torch.float32
            and not embedding_weight.requires_grad and not head_weight.requires_grad
            and (embedding_weight is head_weight or _same_storage(embedding_weight, head_weight))):
        return False

    # Avoid scanning a rejected matrix again on every training step.
    model_core._soup_embed16_checked = True
    if not _is_exact_fp16(embedding_weight):
        return False

    compact_weight = torch.nn.Parameter(embedding_weight.half(), requires_grad=False)
    embedding.weight = compact_weight
    lm_head.weight = compact_weight
    embedding.forward = MethodType(_embedding_forward, embedding)
    del embedding_weight, head_weight
    return True
