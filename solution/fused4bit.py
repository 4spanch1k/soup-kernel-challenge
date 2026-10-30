"""Fused NF4 decode and fp16 matrix multiply for the fixed Soup weight layout."""

import os

import torch
import triton
import triton.language as tl

ENABLED = False
BLOCK_N = 64
BLOCK_K = 64


@triton.jit
def _load_scales(block_ids, absmax_q_ptr, code2_ptr, absmax2_ptr, offset_ptr,
                 valid, SECOND_BLOCK: tl.constexpr):
    codes = tl.load(absmax_q_ptr + block_ids, mask=valid, other=0).to(tl.int32)
    coarse = tl.load(absmax2_ptr + block_ids // SECOND_BLOCK, mask=valid, other=0.0)
    return tl.load(code2_ptr + codes) * coarse + tl.load(offset_ptr)


@triton.jit
def fused4bit_forward_kernel(x_ptr, packed_ptr, absmax_q_ptr, code2_ptr, absmax2_ptr,
                             offset_ptr, nf4_ptr, bias_ptr, y_ptr, M, N, K,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                             BLOCK_K: tl.constexpr, SECOND_BLOCK: tl.constexpr,
                             HAS_BIAS: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_start = tl.program_id(1) * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    k_lanes = tl.arange(0, BLOCK_K)
    byte_lanes = tl.arange(0, BLOCK_K // 2)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + k_lanes
        a_ptrs = x_ptr + rows.to(tl.int64)[:, None] * K + k.to(tl.int64)[None, :]
        a = tl.load(a_ptrs, mask=(rows[:, None] < M) & (k[None, :] < K), other=0.0)
        a = a.to(tl.float16)

        byte_k = k_start // 2 + byte_lanes
        packed_offsets = cols.to(tl.int64)[:, None] * (K // 2) + byte_k.to(tl.int64)[None, :]
        packed = tl.load(packed_ptr + packed_offsets, mask=cols[:, None] < N, other=0)
        packed = packed.to(tl.int32)
        scale_blocks = cols.to(tl.int64) * (K // 64) + k_start // 64
        scales = _load_scales(scale_blocks, absmax_q_ptr, code2_ptr, absmax2_ptr,
                              offset_ptr, cols < N, SECOND_BLOCK)
        high = tl.load(nf4_ptr + (packed >> 4)) * scales[:, None]
        low = tl.load(nf4_ptr + (packed & 15)) * scales[:, None]
        weights = tl.interleave(high, low).to(tl.float16)
        acc = tl.dot(a, tl.trans(weights), acc)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    out_offsets = rows.to(tl.int64)[:, None] * N + cols.to(tl.int64)[None, :]
    tl.store(y_ptr + out_offsets, acc.to(y_ptr.dtype.element_ty),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def fused4bit_input_grad_kernel(dy_ptr, packed_ptr, absmax_q_ptr, code2_ptr, absmax2_ptr,
                                offset_ptr, nf4_ptr, dx_ptr, M, N, K,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                BLOCK_K: tl.constexpr, SECOND_BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_start = tl.program_id(1) * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    reduction_lanes = tl.arange(0, BLOCK_K)
    byte_lanes = tl.arange(0, BLOCK_N // 2)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_K):
        reduction = n_start + reduction_lanes
        dy_offsets = (rows.to(tl.int64)[:, None] * N
                      + reduction.to(tl.int64)[None, :])
        dy = tl.load(dy_ptr + dy_offsets,
                     mask=(rows[:, None] < M) & (reduction[None, :] < N), other=0.0)
        dy = dy.to(tl.float16)

        byte_cols = col_start // 2 + byte_lanes
        packed_offsets = (reduction.to(tl.int64)[:, None] * (K // 2)
                          + byte_cols.to(tl.int64)[None, :])
        packed = tl.load(packed_ptr + packed_offsets,
                         mask=reduction[:, None] < N, other=0).to(tl.int32)
        scale_blocks = reduction.to(tl.int64) * (K // 64) + col_start // 64
        scales = _load_scales(scale_blocks, absmax_q_ptr, code2_ptr, absmax2_ptr,
                              offset_ptr, reduction < N, SECOND_BLOCK)
        high = tl.load(nf4_ptr + (packed >> 4)) * scales[:, None]
        low = tl.load(nf4_ptr + (packed & 15)) * scales[:, None]
        weights = tl.interleave(high, low).to(tl.float16)
        acc = tl.dot(dy, weights, acc)

    out_offsets = rows.to(tl.int64)[:, None] * K + cols.to(tl.int64)[None, :]
    tl.store(dx_ptr + out_offsets, acc.to(dx_ptr.dtype.element_ty),
             mask=(rows[:, None] < M) & (cols[None, :] < K))


def _block_m(m):
    return 32 if m < 512 else 64


def fused_matmul_forward(x, packed, quant_state, bias=None):
    """Compute x @ dequantized_weight.T for a nested NF4 [N, K] weight."""
    n, k = quant_state.shape
    x2 = x.reshape(-1, k).contiguous()
    m = x2.shape[0]
    packed = packed.reshape(-1)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_m = _block_m(m)
    fused4bit_forward_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, BLOCK_N))](
        x2, packed, quant_state.absmax, quant_state.state2.code,
        quant_state.state2.absmax, quant_state.offset.reshape(()), quant_state.code,
        bias if bias is not None else x2, y, m, n, k,
        BLOCK_M=block_m, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SECOND_BLOCK=quant_state.state2.blocksize, HAS_BIAS=bias is not None,
        num_warps=4, num_stages=1)
    return y.view(*x.shape[:-1], n)


def fused_matmul_input_grad(grad_output, packed, quant_state):
    """Compute grad_output @ dequantized_weight for a nested NF4 [N, K] weight."""
    n, k = quant_state.shape
    dy = grad_output.reshape(-1, n).contiguous()
    m = dy.shape[0]
    packed = packed.reshape(-1)
    dx = torch.empty((m, k), dtype=grad_output.dtype, device=grad_output.device)
    block_m = _block_m(m)
    fused4bit_input_grad_kernel[(triton.cdiv(m, block_m), triton.cdiv(k, BLOCK_N))](
        dy, packed, quant_state.absmax, quant_state.state2.code,
        quant_state.state2.absmax, quant_state.offset.reshape(()), quant_state.code,
        dx, m, n, k, BLOCK_M=block_m, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SECOND_BLOCK=quant_state.state2.blocksize, num_warps=4, num_stages=1)
    return dx.view(*grad_output.shape[:-1], k)


def _supported(A, B, out, bias, quant_state):
    try:
        n, k = quant_state.shape
        state2 = quant_state.state2
        packed = B.reshape(-1)
        return (out is None and A.is_cuda and A.dtype == torch.float16 and A.ndim >= 2
                and A.shape[-1] == k and A.stride(-1) == 1 and k % 64 == 0
                and B.is_cuda and B.dtype == torch.uint8 and B.device == A.device
                and packed.numel() * 2 == n * k and quant_state.quant_type == "nf4"
                and quant_state.nested and quant_state.blocksize == 64
                and quant_state.dtype == torch.float16 and quant_state.absmax.dtype == torch.uint8
                and quant_state.absmax.is_contiguous() and quant_state.code.is_contiguous()
                and quant_state.code.dtype == torch.float32 and state2.blocksize == 256
                and state2.code.dtype == torch.float32 and state2.absmax.dtype == torch.float32
                and state2.code.is_contiguous() and state2.absmax.is_contiguous()
                and state2.code.numel() >= 256 and quant_state.code.numel() >= 16
                and all(t.device == A.device for t in
                        (quant_state.absmax, quant_state.code, state2.code,
                         state2.absmax, quant_state.offset))
                and (bias is None or (bias.is_cuda and bias.device == A.device
                                      and bias.dtype == A.dtype and bias.ndim == 1
                                      and bias.stride(0) == 1
                                      and bias.numel() == n)))
    except (AttributeError, TypeError, ValueError):
        return False


def patch_fused4bit():
    """Install the fused path only for the explicit experiment flag."""
    if not ENABLED and os.environ.get("SOUP_FUSED4BIT") != "1":
        return

    from bitsandbytes.autograd import _functions

    cls = _functions.MatMul4Bit
    if getattr(cls.forward, "_soup_fused4bit_patched", False):
        return
    original_forward = cls.forward
    original_backward = cls.backward

    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        ctx._soup_fused4bit = False
        if (quant_state is not None and A.numel() > 0
                and _supported(A, B, out, bias, quant_state)):
            packed = B.reshape(-1, 1)
            output = fused_matmul_forward(A, packed, quant_state, bias)
            ctx.is_empty = False
            ctx.state = quant_state
            ctx.dtype_A, ctx.dtype_B = A.dtype, packed.dtype
            ctx.dtype_bias = None if bias is None else bias.dtype
            ctx.grad_shape = A.shape
            ctx.tensors = (None, packed) if any(ctx.needs_input_grad[:2]) else (None, None)
            ctx._soup_fused4bit = True
            return output
        return original_forward(ctx, A, B, out, bias, quant_state)

    def backward(ctx, grad_output):
        if not getattr(ctx, "_soup_fused4bit", False):
            return original_backward(ctx, grad_output)

        req_grad_a, _, _, req_grad_bias, _ = ctx.needs_input_grad
        _, packed = ctx.tensors
        grad_a = None
        grad_bias = None
        if req_grad_bias:
            n = ctx.state.shape[0]
            grad_bias = grad_output.reshape(-1, n).sum(0, dtype=ctx.dtype_bias)
        if req_grad_a:
            grad_a = fused_matmul_input_grad(grad_output, packed, ctx.state).view(ctx.grad_shape)
        return grad_a, None, None, grad_bias, None

    forward._soup_fused4bit_patched = True
    cls.forward = staticmethod(forward)
    cls.backward = staticmethod(backward)
