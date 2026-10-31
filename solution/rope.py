"""Fused rotary position embedding for Qwen2, forward and backward in Triton.

`apply_rotary_pos_emb` builds `q * cos + rotate_half(q) * sin` from a slice, a negation, a cat,
two multiplications and an add, for q and for k, and keeps all of it for backward. One program
here handles one (batch, position, head): it reads the two halves of the head vector and writes
both rotated halves. Backward is the same rotation by the opposite angle.

Under fp16 autocast the PyTorch version returns fp32 (cos/sin are fp32) and scaled_dot_product_
attention then casts back to fp16; with autocast on this kernel writes the input dtype directly,
which is the same value rounded at the same place and half the memory.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, seq, sxb, sxh, sxt, sob, soh, sot, scb, sct,
                HALF: tl.constexpr, BACKWARD: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    batch = (token // seq).to(tl.int64)
    pos = (token % seq).to(tl.int64)
    offs = tl.arange(0, HALF)
    src = x_ptr + batch * sxb + head * sxh + pos * sxt
    dst = out_ptr + batch * sob + head * soh + pos * sot
    table = batch * scb + pos * sct
    x1 = tl.load(src + offs).to(tl.float32)
    x2 = tl.load(src + HALF + offs).to(tl.float32)
    c1 = tl.load(cos_ptr + table + offs).to(tl.float32)
    c2 = tl.load(cos_ptr + table + HALF + offs).to(tl.float32)
    s1 = tl.load(sin_ptr + table + offs).to(tl.float32)
    s2 = tl.load(sin_ptr + table + HALF + offs).to(tl.float32)
    if BACKWARD:
        out1 = x1 * c1 + x2 * s2
        out2 = x2 * c2 - x1 * s1
    else:
        out1 = x1 * c1 - x2 * s1
        out2 = x2 * c2 + x1 * s2
    tl.store(dst + offs, out1.to(out_ptr.dtype.element_ty))
    tl.store(dst + HALF + offs, out2.to(out_ptr.dtype.element_ty))


if os.environ.get("SOUP_X_MERGE_VARIANTS") == "1":
    rope_kernel = triton.jit(do_not_specialize=["seq"])(rope_kernel.fn)


def _launch(x, cos, sin, out_dtype, backward):
    batch, heads, seq, dim = x.shape
    out = torch.empty_like(x, dtype=out_dtype)
    rope_kernel[(batch * seq, heads)](
        x, cos, sin, out, seq, x.stride(0), x.stride(1), x.stride(2), out.stride(0),
        out.stride(1), out.stride(2), cos.stride(0) if cos.shape[0] > 1 else 0, cos.stride(1),
        HALF=dim // 2,
        BACKWARD=backward, num_warps=2)
    return out


class RotaryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, out_dtype):
        ctx.save_for_backward(cos, sin)
        ctx.x_dtype = x.dtype
        return _launch(x, cos, sin, out_dtype, False)

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        if grad.stride(-1) != 1:
            grad = grad.contiguous()
        return _launch(grad, cos, sin, ctx.x_dtype, True), None, None, None


def _supported(x, cos, sin):
    dim = x.shape[-1]
    return (x.is_cuda and x.dim() == 4 and x.stride(-1) == 1 and cos.dim() == 3
            and cos.shape == sin.shape and cos.stride() == sin.stride() and cos.stride(-1) == 1
            and cos.shape[0] in (1, x.shape[0]) and cos.shape[1] == x.shape[2] and cos.shape[2] == dim
            and dim % 2 == 0 and (dim // 2) & (dim // 2 - 1) == 0 and not cos.requires_grad
            and x.dtype in (torch.float16, torch.float32))


def _autocast_on():
    try:
        return torch.is_autocast_enabled("cuda")
    except TypeError:  # older torch: no device argument
        return torch.is_autocast_enabled()


def patch_qwen2_rope():
    """Replace the module-level `apply_rotary_pos_emb` that Qwen2Attention.forward looks up."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original, "_soup_rope_patched", False):
        return

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if unsqueeze_dim != 1 or not (_supported(q, cos, sin) and _supported(k, cos, sin)):
            return original(q, k, cos, sin, unsqueeze_dim)
        out_q = q.dtype if _autocast_on() else torch.promote_types(q.dtype, cos.dtype)
        out_k = k.dtype if _autocast_on() else torch.promote_types(k.dtype, cos.dtype)
        return (RotaryFunction.apply(q, cos, sin, out_q), RotaryFunction.apply(k, cos, sin, out_k))

    apply_rotary_pos_emb._soup_rope_patched = True
    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb
