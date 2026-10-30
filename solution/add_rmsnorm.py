"""Fused residual add + RMSNorm for the MLP half of a Qwen2 decoder layer, in Triton.

A Qwen2 layer does `h = residual + attn_out` and then `post_attention_layernorm(h)`. In our run the
residual stream is fp32 (the embedding is kept in fp32), `attn_out` is fp16 (autocast), and the
MLP linears cast the fp32 norm output to fp16 again and again. PyTorch runs this as an add
kernel, the norm in several kernels and one cast per consumer. Here one kernel reads the residual
and the branch output once, writes the new residual (fp32, as before) and the normalised value
directly in the dtype the MLP consumes (fp16 under autocast), and keeps only `rstd` for backward.

Backward is one kernel as well: it returns the gradient of the sum to both addends, the norm's
part plus the gradient that arrives through the residual path.

Numerics follow the PyTorch code: the sum is rounded to the residual dtype first and the norm
works on that value; autocast would round the fp32 norm output to fp16 for every linear, here the
same rounding happens once in the kernel.
"""

import torch
import triton
import triton.language as tl

from .amp import autocast_dtype

MAX_WIDTH = 16384


@triton.jit
def add_rmsnorm_fwd_kernel(res_ptr, delta_ptr, w_ptr, sum_ptr, y_ptr, rstd_ptr, width, eps,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < width
    offsets = row * width + cols
    total = (tl.load(res_ptr + offsets, mask=inside, other=0.0).to(tl.float32)
             + tl.load(delta_ptr + offsets, mask=inside, other=0.0).to(tl.float32))
    stored = total.to(sum_ptr.dtype.element_ty)
    tl.store(sum_ptr + offsets, stored, mask=inside)
    x = stored.to(tl.float32)  # the norm sees the rounded sum, as it does in PyTorch
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, 0) / width + eps)
    normed = (x * rstd).to(sum_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    tl.store(y_ptr + offsets, (normed * w).to(y_ptr.dtype.element_ty), mask=inside)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def add_rmsnorm_bwd_kernel(dy_ptr, dsum_ptr, sum_ptr, w_ptr, rstd_ptr, dres_ptr, ddelta_ptr, width,
                           BLOCK: tl.constexpr, HAS_DSUM: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < width
    offsets = row * width + cols
    dy = tl.load(dy_ptr + offsets, mask=inside, other=0.0).to(tl.float32)
    x = tl.load(sum_ptr + offsets, mask=inside, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    xhat = x * rstd
    wdy = dy * w
    grad = rstd * (wdy - xhat * (tl.sum(wdy * xhat, 0) / width))
    if HAS_DSUM:  # the residual path of the sum
        grad += tl.load(dsum_ptr + offsets, mask=inside, other=0.0).to(tl.float32)
    tl.store(dres_ptr + offsets, grad.to(dres_ptr.dtype.element_ty), mask=inside)
    tl.store(ddelta_ptr + offsets, grad.to(ddelta_ptr.dtype.element_ty), mask=inside)


class AddRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, delta, weight, eps, y_dtype):
        shape = residual.shape
        width = shape[-1]
        res = residual.reshape(-1, width).contiguous()
        dlt = delta.reshape(-1, width).contiguous()
        rows = res.shape[0]
        total = torch.empty_like(res, dtype=torch.promote_types(res.dtype, dlt.dtype))
        y = torch.empty(rows, width, dtype=y_dtype, device=res.device)
        rstd = torch.empty(rows, dtype=torch.float32, device=res.device)
        add_rmsnorm_fwd_kernel[(rows,)](res, dlt, weight, total, y, rstd, width, eps,
                                        BLOCK=triton.next_power_of_2(width), num_warps=8)
        ctx.save_for_backward(total, weight, rstd)
        ctx.shape, ctx.res_dtype, ctx.delta_dtype = shape, residual.dtype, delta.dtype
        return y.view(shape), total.view(shape)

    @staticmethod
    def backward(ctx, dy, dsum):
        total, weight, rstd = ctx.saved_tensors
        rows, width = total.shape
        dy2 = dy.reshape(rows, width).contiguous()
        has_dsum = dsum is not None
        dsum2 = dsum.reshape(rows, width).contiguous() if has_dsum else dy2
        dres = torch.empty(rows, width, dtype=ctx.res_dtype, device=total.device)
        ddelta = torch.empty(rows, width, dtype=ctx.delta_dtype, device=total.device)
        add_rmsnorm_bwd_kernel[(rows,)](dy2, dsum2, total, weight, rstd, dres, ddelta, width,
                                        BLOCK=triton.next_power_of_2(width),
                                        HAS_DSUM=has_dsum, num_warps=8)
        dweight = None
        if ctx.needs_input_grad[2]:  # the norms are frozen in this task; kept for correctness
            normed = (total.float() * rstd[:, None]).to(total.dtype)
            dweight = (dy2.float() * normed.float()).sum(0).to(weight.dtype)
        return dres.view(ctx.shape), ddelta.view(ctx.shape), dweight, None, None


def add_rmsnorm(residual, delta, weight, eps, y_dtype):
    """Returns (rmsnorm(residual + delta) * weight in y_dtype, residual + delta)."""
    return AddRMSNormFunction.apply(residual, delta, weight, eps, y_dtype)


def patch_qwen2_decoder_layer():
    """Qwen2DecoderLayer.forward with the add + post-attention norm fused. Training only: eval,
    CPU, another norm class or an unexpected dtype take the original code."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2DecoderLayer
    if getattr(cls.forward, "_soup_add_rmsnorm_patched", False):
        return
    original = cls.forward

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, position_embeddings=None, **kwargs):
        norm = self.post_attention_layernorm
        if not (self.training and torch.is_grad_enabled() and hidden_states.is_cuda
                and type(norm) is modeling_qwen2.Qwen2RMSNorm and norm.weight.is_cuda
                and hidden_states.shape[-1] <= MAX_WIDTH
                and hidden_states.dtype in (torch.float16, torch.float32)):
            return original(self, hidden_states, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            position_embeddings=position_embeddings, **kwargs)
        y_dtype = autocast_dtype() or torch.promote_types(residual.dtype, norm.weight.dtype)
        normed, hidden_states = add_rmsnorm(residual, attn_out, norm.weight,
                                            norm.variance_epsilon, y_dtype)
        return hidden_states + self.mlp(normed)

    forward._soup_add_rmsnorm_patched = True
    cls.forward = forward
