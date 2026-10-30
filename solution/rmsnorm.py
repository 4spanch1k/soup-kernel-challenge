"""Fused RMSNorm for Qwen2, forward and backward in Triton.

Qwen2RMSNorm runs as about seven PyTorch ops (cast to fp32, pow, mean, add eps, rsqrt, mul, cast,
mul by the weight), each reading and writing the whole activation, and autograd keeps several
fp32 intermediates for backward. Here a row is read once, normalised in registers and written
once; backward keeps only the input (already alive as the residual) and one fp32 `rstd` per row.

Numerics follow the PyTorch module: statistics in fp32, the normalised value rounded to the input
dtype before the multiplication by the weight.

Output dtype: PyTorch returns promote(input, weight) = fp32 here (the residual stream is fp32), and
every consumer then casts it to fp16 under autocast. Worse, bitsandbytes' Linear4bit returns its
result in the dtype of its input, so each 4-bit linear used to convert its fp16 result back to
fp32 ([tokens, 8960] copies in the MLP) and the saved activations were twice as big. With autocast
on this kernel writes the autocast dtype directly: the same value rounded once at the same place.
"""

import torch
import triton
import triton.language as tl

from .amp import autocast_dtype

MAX_WIDTH = 16384  # one program holds a whole row


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, width, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < width
    x = tl.load(x_ptr + row * width + cols, mask=inside, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, 0) / width + eps)
    normed = (x * rstd).to(x_ptr.dtype.element_ty).to(tl.float32)  # rounded like .to(input_dtype)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * width + cols, (normed * w).to(y_ptr.dtype.element_ty), mask=inside)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, width, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < width
    dy = tl.load(dy_ptr + row * width + cols, mask=inside, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * width + cols, mask=inside, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    xhat = x * rstd
    wdy = dy * w
    mean_term = tl.sum(wdy * xhat, 0) / width
    dx = rstd * (wdy - xhat * mean_term)
    tl.store(dx_ptr + row * width + cols, dx.to(dx_ptr.dtype.element_ty), mask=inside)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, out_dtype=None):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        rows, width = x2.shape
        y = torch.empty(rows, width, dtype=out_dtype or torch.promote_types(x.dtype, weight.dtype),
                        device=x.device)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        rmsnorm_fwd_kernel[(rows,)](x2, weight, y, rstd, width, eps,
                                    BLOCK=triton.next_power_of_2(width), num_warps=8)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape, ctx.x_dtype = shape, x.dtype
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        rows, width = x2.shape
        dy2 = dy.reshape(rows, width).contiguous()
        dx = torch.empty_like(x2)
        rmsnorm_bwd_kernel[(rows,)](dy2, x2, weight, rstd, dx, width,
                                    BLOCK=triton.next_power_of_2(width), num_warps=8)
        dweight = None
        if ctx.needs_input_grad[1]:  # the norms are frozen in this task; kept for correctness
            normed = (x2.float() * rstd[:, None]).to(ctx.x_dtype)
            dweight = (dy2.float() * normed.float()).sum(0).to(weight.dtype)
        return dx.view(ctx.shape), dweight, None, None


def rmsnorm(x, weight, eps, out_dtype=None):
    return RMSNormFunction.apply(x, weight, eps, out_dtype)


def patch_qwen2_rmsnorm():
    """Send Qwen2RMSNorm.forward through the fused kernel (a class patch: Python looks the method
    up on every call, so it works before or after Soup builds the model)."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls.forward, "_soup_rmsnorm_patched", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        if (not hidden_states.is_cuda or not self.weight.is_cuda
                or hidden_states.shape[-1] > MAX_WIDTH or hidden_states.dtype not in
                (torch.float16, torch.float32)):
            return original(self, hidden_states)
        return rmsnorm(hidden_states, self.weight, self.variance_epsilon, autocast_dtype())

    forward._soup_rmsnorm_patched = True
    cls.forward = forward
