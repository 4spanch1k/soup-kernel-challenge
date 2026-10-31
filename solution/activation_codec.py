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
    if os.environ.get('SOUP_X_ACT_INT8') != '1':
        return
    from transformers.models.qwen2 import modeling_qwen2
    cls = modeling_qwen2.Qwen2MLP
    original = cls.forward
    if getattr(original, '_soup_int8_saved', False):
        return

    def forward(self, x):
        if (not self.training or not torch.is_grad_enabled() or not x.is_cuda
                or not x.requires_grad
                or getattr(getattr(self, 'config', None), 'hidden_act', 'silu') != 'silu'):
            return original(self, x)
        # Task MLP weights are frozen; preserve every unexpected trainable case.
        if any(parameter.requires_grad for parameter in self.parameters()):
            return original(self, x)
        return self.down_proj(SavedActivationInt8.apply(self.gate_proj(x), self.up_proj(x)))

    forward._soup_int8_saved = True
    cls.forward = forward
