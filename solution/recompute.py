"""Own recomputation of the Qwen2 MLP: trade a little time for a lot of activation memory.

Without it autograd keeps gate, up and the activation of every MLP (about 3 x 2048 x 8960 fp16
values per layer, 28 layers). Here the MLP is wrapped in `torch.utils.checkpoint`: only its input
is kept and the MLP runs once more inside backward. This is our own code in `apply()`, not Soup's
`gradient_checkpointing` switch (which the rules forbid).

OFF by default: the extra forward costs time (est. +5..8% of a step) and the best trade between
the speed and the memory score is decided by measurement, not here. Turn it on by setting
`ENABLED = True` (or `SOUP_RECOMPUTE_MLP=1` while experimenting on Kaggle).
"""

import os

import torch
from torch.utils.checkpoint import checkpoint

ENABLED = False


def patch_qwen2_mlp_recompute(force=None):
    """Call after patch_qwen2_mlp(): it wraps whichever Qwen2MLP.forward is installed then."""
    enabled = ENABLED or os.environ.get("SOUP_RECOMPUTE_MLP") == "1" if force is None else force
    if not enabled:
        return
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls.forward, "_soup_recompute_patched", False):
        return
    inner = cls.forward

    def forward(self, x):
        if not (self.training and torch.is_grad_enabled() and x.requires_grad and x.is_cuda):
            return inner(self, x)
        return checkpoint(inner, self, x, use_reentrant=False)

    forward._soup_recompute_patched = True
    cls.forward = forward


def patch_qwen2_mlp_adaptive_recompute():
    """Experiment (off by default): recompute the MLP only when a batch carries many real tokens.

    The memory peak of a run is set by its densest batches only. With `SOUP_X_ADAPT_RECOMP=<rows>` the
    MLP of a forward pass is wrapped in our own `torch.utils.checkpoint` only if that pass handles at
    least `<rows>` token rows (valid tokens in the packed path, batch x length in the padded path), so
    the extra forward is paid on a few steps instead of every step. Call after patch_qwen2_mlp().
    """
    try:
        threshold = int(os.environ.get("SOUP_X_ADAPT_RECOMP", "0"))
    except ValueError:
        return
    if threshold <= 0:
        return
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls.forward, "_soup_adapt_recompute_patched", False):
        return
    inner = cls.forward

    def forward(self, x):
        if not (self.training and torch.is_grad_enabled() and x.requires_grad and x.is_cuda):
            return inner(self, x)
        rows = x.numel() // max(x.shape[-1], 1)
        if rows < threshold:
            return inner(self, x)
        return checkpoint(inner, self, x, use_reentrant=False)

    forward._soup_adapt_recompute_patched = True
    cls.forward = forward
