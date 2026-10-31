"""Opt-in int8 saved activations for the MLP; forward retains full fp16 arithmetic.

Only gate/up tensors saved for backward are approximated. We do not quantize the
MLP output or the frozen model weights. Int8 is a storage format, not int8 GEMM.
Idea: blockwise saved-activation compression (HyC-LoRA; Team La's public report
also applies saved-activation compression). This implementation is independent.
"""
import os
import torch
import triton
import triton.language as tl

BLOCK = 1024
INT4_BLOCK = 128


@triton.jit
def _int4_encode(value, scale):
    # Sign + three magnitude bits. More levels near zero protect SiLU's slope;
    # thresholds are midpoints of our independent, symmetric magnitude table.
    magnitude = tl.abs(value) / scale
    code = ((magnitude > .035).to(tl.int32) + (magnitude > .11).to(tl.int32)
            + (magnitude > .20).to(tl.int32) + (magnitude > .31).to(tl.int32)
            + (magnitude > .445).to(tl.int32) + (magnitude > .62).to(tl.int32)
            + (magnitude > .86).to(tl.int32))
    return code | tl.where(value < 0., 8, 0)


@triton.jit
def _int4_decode(code, scale):
    magnitude = code & 7
    level = tl.where(magnitude == 0, 0., tl.where(magnitude == 1, .07,
            tl.where(magnitude == 2, .15, tl.where(magnitude == 3, .25,
            tl.where(magnitude == 4, .37, tl.where(magnitude == 5, .52,
            tl.where(magnitude == 6, .72, 1.)))))))
    return tl.where((code & 8) != 0, -level, level) * scale


@triton.jit
def activation_int4_pack_kernel(gate, up, result, gate_codes, up_codes,
                               gate_scales, up_scales, count, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    idx = block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < count
    g = tl.load(gate + idx, valid, 0.).to(tl.float32)
    u = tl.load(up + idx, valid, 0.).to(tl.float32)
    gs = tl.maximum(tl.max(tl.abs(g), 0), 1.e-8)
    us = tl.maximum(tl.max(tl.abs(u), 0), 1.e-8)
    gc = _int4_encode(g, gs)
    uc = _int4_encode(u, us)
    g0, g1 = tl.split(tl.reshape(gc, (BLOCK // 2, 2)))
    u0, u1 = tl.split(tl.reshape(uc, (BLOCK // 2, 2)))
    byte_idx = block * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    tl.store(gate_codes + byte_idx, (g0 | (g1 << 4)).to(tl.uint8), byte_idx * 2 < count)
    tl.store(up_codes + byte_idx, (u0 | (u1 << 4)).to(tl.uint8), byte_idx * 2 < count)
    # Forward always uses the original values, never the decoded approximation.
    exact_output = (g / (1. + tl.exp(-g))) * u
    tl.store(result + idx, exact_output.to(result.dtype.element_ty), valid)
    tl.store(gate_scales + block, gs)
    tl.store(up_scales + block, us)


@triton.jit
def activation_int4_unpack_backward_kernel(dout, gate_codes, up_codes,
                                          gate_scales, up_scales, dgate, dup,
                                          count, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    idx = block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < count
    shift = (idx & 1) * 4
    gc = (tl.load(gate_codes + idx // 2, valid, 0).to(tl.int32) >> shift) & 15
    uc = (tl.load(up_codes + idx // 2, valid, 0).to(tl.int32) >> shift) & 15
    g = _int4_decode(gc, tl.load(gate_scales + block))
    u = _int4_decode(uc, tl.load(up_scales + block))
    grad = tl.load(dout + idx, valid, 0.).to(tl.float32)
    sigmoid = 1. / (1. + tl.exp(-g))
    activation = g * sigmoid
    derivative = sigmoid + activation * (1. - sigmoid)
    tl.store(dgate + idx, (grad * u * derivative).to(dgate.dtype.element_ty), valid)
    tl.store(dup + idx, (grad * activation).to(dup.dtype.element_ty), valid)


class SavedActivationInt4(torch.autograd.Function):
    """Two four-bit codes per byte, with separate fp32 scales per 128 values.

    The approximation affects only frozen-MLP backward. The 15 distinct signed
    levels include exact zero; the duplicated negative-zero code is harmless.
    """

    @staticmethod
    def forward(ctx, gate, up):
        g, u = gate.contiguous(), up.to(gate.dtype).contiguous()
        out = torch.empty_like(g)
        codes = triton.cdiv(g.numel(), 2)
        gc = torch.empty(codes, device=g.device, dtype=torch.uint8)
        uc = torch.empty_like(gc)
        blocks = triton.cdiv(g.numel(), INT4_BLOCK)
        gs = torch.empty(blocks, device=g.device, dtype=torch.float32)
        us = torch.empty_like(gs)
        activation_int4_pack_kernel[(blocks,)](
            g, u, out, gc, uc, gs, us, g.numel(), BLOCK=INT4_BLOCK, num_warps=4)
        ctx.save_for_backward(gc, uc, gs, us)
        ctx.shape, ctx.dtype, ctx.count = g.shape, g.dtype, g.numel()
        return out

    @staticmethod
    def backward(ctx, grad):
        gc, uc, gs, us = ctx.saved_tensors
        grad = grad.to(ctx.dtype).contiguous()
        dg = torch.empty(ctx.shape, device=grad.device, dtype=ctx.dtype)
        du = torch.empty_like(dg)
        activation_int4_unpack_backward_kernel[(triton.cdiv(ctx.count, INT4_BLOCK),)](
            grad, gc, uc, gs, us, dg, du, ctx.count, BLOCK=INT4_BLOCK, num_warps=4)
        return dg, du


@triton.jit
def activation_pack_kernel(gate, up, result, gate_codes, up_codes, gate_scales, up_scales,
                           count, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    idx = block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < count
    g = tl.load(gate + idx, valid, 0.).to(tl.float32)
    u = tl.load(up + idx, valid, 0.).to(tl.float32)
    gs = tl.maximum(tl.max(tl.abs(g), 0) / 127., 1.e-8)
    us = tl.maximum(tl.max(tl.abs(u), 0) / 127., 1.e-8)
    gc = tl.minimum(127., tl.maximum(-127., tl.floor(g / gs + .5))).to(tl.int8)
    uc = tl.minimum(127., tl.maximum(-127., tl.floor(u / us + .5))).to(tl.int8)
    exact_output = (g / (1. + tl.exp(-g))) * u
    tl.store(result + idx, exact_output.to(result.dtype.element_ty), valid)
    tl.store(gate_codes + idx, gc, valid)
    tl.store(up_codes + idx, uc, valid)
    tl.store(gate_scales + block, gs)
    tl.store(up_scales + block, us)


@triton.jit
def activation_unpack_backward_kernel(dout, gate_codes, up_codes, gate_scales, up_scales,
                                      dgate, dup, count, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    idx = block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < count
    g = tl.load(gate_codes + idx, valid, 0).to(tl.float32) * tl.load(gate_scales + block)
    u = tl.load(up_codes + idx, valid, 0).to(tl.float32) * tl.load(up_scales + block)
    grad = tl.load(dout + idx, valid, 0.).to(tl.float32)
    sig = 1. / (1. + tl.exp(-g))
    activation = g * sig
    derivative = sig + activation * (1. - sig)
    tl.store(dgate + idx, (grad * u * derivative).to(dgate.dtype.element_ty), valid)
    tl.store(dup + idx, (grad * activation).to(dup.dtype.element_ty), valid)


class SavedActivationInt8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        g, u = gate.contiguous(), up.to(gate.dtype).contiguous()
        out = torch.empty_like(g)
        gc, uc = torch.empty_like(g, dtype=torch.int8), torch.empty_like(u, dtype=torch.int8)
        blocks = triton.cdiv(g.numel(), BLOCK)
        gs = torch.empty(blocks, device=g.device, dtype=torch.float32)
        us = torch.empty_like(gs)
        activation_pack_kernel[(blocks,)](g, u, out, gc, uc, gs, us, g.numel(),
                                          BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(gc, uc, gs, us)
        ctx.dtype = g.dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        gc, uc, gs, us = ctx.saved_tensors
        grad = grad.to(ctx.dtype).contiguous()
        dg = torch.empty_like(gc, dtype=ctx.dtype)
        du = torch.empty_like(uc, dtype=ctx.dtype)
        activation_unpack_backward_kernel[(triton.cdiv(gc.numel(), BLOCK),)](
            grad, gc, uc, gs, us, dg, du, gc.numel(), BLOCK=BLOCK, num_warps=4)
        return dg, du


def patch_saved_activation_int8():
    int4 = os.environ.get('SOUP_X_ACT_INT4') == '1'
    if not int4 and os.environ.get('SOUP_X_ACT_INT8') != '1':
        return
    from transformers.models.qwen2 import modeling_qwen2
    cls = modeling_qwen2.Qwen2MLP
    original = cls.forward
    mode = 'int4' if int4 else 'int8'
    if getattr(original, '_soup_saved_codec', None) == mode:
        return
    original = getattr(original, '_soup_activation_original', original)
    codec = SavedActivationInt4 if int4 else SavedActivationInt8

    def forward(self, x):
        if (not self.training or not torch.is_grad_enabled() or not x.is_cuda
                or not x.requires_grad
                or getattr(getattr(self, 'config', None), 'hidden_act', 'silu') != 'silu'):
            return original(self, x)
        # Task MLP weights are frozen; preserve every unexpected trainable case.
        if any(parameter.requires_grad for parameter in self.parameters()):
            return original(self, x)
        return self.down_proj(codec.apply(self.gate_proj(x), self.up_proj(x)))

    forward._soup_int8_saved = True
    forward._soup_saved_codec = mode
    forward._soup_activation_original = original
    cls.forward = forward
