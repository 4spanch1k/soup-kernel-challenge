"""Fuse LoRA dropout and output addition for CUDA linear layers."""

import os

import torch
import triton
import triton.language as tl

from .amp import autocast_dtype

BLOCK = 1024
MAX_RANK = 64
MAX_WIDTH = 16384
ENABLED = False


@triton.jit
def _dropout_cast_kernel(x_ptr, seed_ptr, out_ptr, n_elements, p, scale,
                         BLOCK: tl.constexpr, DROPOUT: tl.constexpr,
                         ADAPTER_FP16: tl.constexpr, COMPUTE_FP16: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if ADAPTER_FP16:
        values = values.to(tl.float16).to(tl.float32)
    if DROPOUT:
        seed = tl.load(seed_ptr).to(tl.int32)
        keep = tl.rand(seed, offsets) >= p
        values = tl.where(keep, values * scale, 0.0)
    if COMPUTE_FP16:
        values = values.to(tl.float16)
    tl.store(out_ptr + offsets, values, mask=mask)


@triton.jit
def _dropout_backward_kernel(grad_ptr, seed_ptr, grad_x_ptr, n_elements, p, scale,
                             BLOCK: tl.constexpr, DROPOUT: tl.constexpr,
                             ADAPTER_FP16: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if ADAPTER_FP16:
        grad = grad.to(tl.float16).to(tl.float32)
    if DROPOUT:
        seed = tl.load(seed_ptr).to(tl.int32)
        keep = tl.rand(seed, offsets) >= p
        grad = tl.where(keep, grad * scale, 0.0)
        if ADAPTER_FP16:
            grad = grad.to(tl.float16)
    tl.store(grad_x_ptr + offsets, grad, mask=mask)


@triton.jit
def _add_scaled_kernel(base_ptr, lora_ptr, out_ptr, n_elements, scale,
                       BLOCK: tl.constexpr, COMPUTE_FP16: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    base = tl.load(base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    lora = tl.load(lora_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    lora = lora * scale
    if COMPUTE_FP16:
        lora = lora.to(tl.float16).to(tl.float32)
    tl.store(out_ptr + offsets, base + lora, mask=mask)


def _dropout_cast(x, seed, p, scale, adapter_fp16, compute_fp16):
    dropped = torch.empty(x.shape, device=x.device,
                          dtype=torch.float16 if compute_fp16 else torch.float32)
    _dropout_cast_kernel[(triton.cdiv(x.numel(), BLOCK),)](
        x, seed, dropped, x.numel(), p, scale, BLOCK=BLOCK, DROPOUT=p > 0.0,
        ADAPTER_FP16=adapter_fp16, COMPUTE_FP16=compute_fp16, num_warps=4)
    return dropped


class LoRAFuseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, base_out, a, b, scaling, p, seed, compute_fp16, adapter_fp16):
        n, in_features = x.numel() // x.shape[-1], x.shape[-1]
        out_features = base_out.shape[-1]
        x2 = x.reshape(n, in_features).contiguous()
        base2 = base_out.reshape(n, out_features).contiguous()
        compute_dtype = torch.float16 if compute_fp16 else torch.float32
        a_compute = a.to(compute_dtype).contiguous()
        b_compute = b.to(compute_dtype).contiguous()
        dropped = _dropout_cast(x2, seed, p, 1.0 / (1.0 - p), adapter_fp16, compute_fp16)
        with torch.autocast("cuda", enabled=False):
            intermediate = torch.mm(dropped, a_compute.t())
            lora_out = torch.mm(intermediate, b_compute.t())
        out = torch.empty_like(base2)
        _add_scaled_kernel[(triton.cdiv(base2.numel(), BLOCK),)](
            base2, lora_out, out, base2.numel(), scaling, BLOCK=BLOCK,
            COMPUTE_FP16=compute_fp16, num_warps=4)

        ctx.save_for_backward(x2, intermediate, seed, a_compute, b_compute)
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        ctx.base_dtype = base_out.dtype
        ctx.adapter_dtype = a.dtype
        ctx.scaling, ctx.p = scaling, p
        ctx.compute_fp16, ctx.adapter_fp16 = compute_fp16, adapter_fp16
        return out.view(base_out.shape)

    @staticmethod
    def backward(ctx, grad_out):
        x, intermediate, seed, a_compute, b_compute = ctx.saved_tensors
        n, in_features = x.shape
        out_features = b_compute.shape[0]
        grad_out2 = grad_out.reshape(n, out_features).contiguous()
        grad_base = grad_out.to(ctx.base_dtype) if ctx.needs_input_grad[1] else None
        need_x = ctx.needs_input_grad[0]
        need_a = ctx.needs_input_grad[2]
        need_b = ctx.needs_input_grad[3]
        grad_x = grad_a = grad_b = None

        if need_x or need_a or need_b:
            compute_dtype = torch.float16 if ctx.compute_fp16 else torch.float32
            with torch.autocast("cuda", enabled=False):
                grad_lora = grad_out2.to(compute_dtype) * ctx.scaling
                grad_z = torch.mm(grad_lora, b_compute) if need_x or need_a else None
                if need_b:
                    grad_b = torch.mm(grad_lora.t(), intermediate).to(ctx.adapter_dtype)
                if need_a:
                    dropped = _dropout_cast(x, seed, ctx.p, 1.0 / (1.0 - ctx.p),
                                            ctx.adapter_fp16, ctx.compute_fp16)
                    grad_a = torch.mm(grad_z.t(), dropped).to(ctx.adapter_dtype)
                if need_x:
                    grad_x2 = torch.empty_like(x)
                    grad_input = torch.mm(grad_z, a_compute)
                    _dropout_backward_kernel[(triton.cdiv(x.numel(), BLOCK),)](
                        grad_input, seed, grad_x2, x.numel(), ctx.p, 1.0 / (1.0 - ctx.p),
                        BLOCK=BLOCK, DROPOUT=ctx.p > 0.0,
                        ADAPTER_FP16=ctx.adapter_fp16, num_warps=4)
                    grad_x = grad_x2.view(ctx.x_shape)

        return grad_x, grad_base, grad_a, grad_b, None, None, None, None, None


def lora_fused(x, base_out, a, b, scaling, p, seed=None):
    """Add one LoRA branch to a linear output, regenerating dropout in backward."""
    if seed is None:
        if p > 0.0:
            seed = torch.randint(0, 2**31 - 1, (), device=x.device, dtype=torch.int64)
        else:
            seed = torch.empty(0, device=x.device, dtype=torch.int64)
    adapter_fp16 = a.dtype == torch.float16
    compute_dtype = autocast_dtype() or a.dtype
    return LoRAFuseFunction.apply(x, base_out, a, b, scaling, p, seed,
                                  compute_dtype == torch.float16, adapter_fp16)


def _base_returns_fp16(base, x):
    dtype = autocast_dtype()
    if isinstance(base, torch.nn.Linear):
        return dtype == torch.float16 or base.weight.dtype == torch.float16
    return (type(base).__name__ == "Linear4bit"
            and type(base).__module__.startswith("bitsandbytes")
            and x.dtype == torch.float16)


def _can_fuse(module, x, args, kwargs):
    if (args or kwargs or not module.training or not torch.is_grad_enabled()
            or module.disable_adapters or module.merged or module.merged_adapters
            or not module.cast_input_dtype_enabled or x.dim() not in (2, 3)
            or not x.is_cuda or not x.is_contiguous()
            or x.dtype not in (torch.float16, torch.float32)):
        return None
    active = module.active_adapters
    if len(active) != 1:
        return None
    adapter = active[0]
    if (adapter not in module.lora_A or adapter not in module.lora_B
            or module.use_dora.get(adapter, False) or module.lora_bias.get(adapter, False)
            or adapter in module.lora_variant):
        return None
    dropout = module.lora_dropout[adapter]
    if (type(dropout) not in (torch.nn.Dropout, torch.nn.Identity)
            or isinstance(dropout, torch.nn.Dropout) and dropout.inplace):
        return None
    p = dropout.p if isinstance(dropout, torch.nn.Dropout) and dropout.training else 0.0
    if p >= 1.0:
        return None
    a, b = module.lora_A[adapter], module.lora_B[adapter]
    if (a.bias is not None or b.bias is not None or a.weight.dtype not in
            (torch.float16, torch.float32) or b.weight.dtype != a.weight.dtype
            or not a.weight.is_cuda or not b.weight.is_cuda
            or a.weight.device != x.device or b.weight.device != x.device
            or not a.weight.is_contiguous() or not b.weight.is_contiguous()):
        return None
    rank, in_features = a.weight.shape
    out_features = b.weight.shape[0]
    if (rank < 1 or rank > MAX_RANK or in_features > MAX_WIDTH or out_features > MAX_WIDTH
            or a.weight.shape[1] != x.shape[-1] or b.weight.shape[1] != rank
            or module.r[adapter] != rank or not _base_returns_fp16(module.get_base_layer(), x)):
        return None
    compute_dtype = autocast_dtype()
    if compute_dtype not in (None, torch.float16):
        return None
    return adapter, p


def patch_lora_fuse():
    """Patch PEFT's Linear only when explicitly enabled for the T4 experiment."""
    if not (ENABLED or os.environ.get("SOUP_LORA_FUSE") == "1"):
        return
    from peft.tuners.lora.layer import Linear

    if getattr(Linear.forward, "_soup_lora_fuse_patched", False):
        return
    original = Linear.forward

    def forward(self, x, *args, **kwargs):
        eligible = _can_fuse(self, x, args, kwargs)
        if eligible is None:
            return original(self, x, *args, **kwargs)
        adapter, p = eligible
        base_out = self.base_layer(x, *args, **kwargs)
        if base_out.dtype != torch.float16:
            return original(self, x, *args, **kwargs)
        return lora_fused(x, base_out, self.lora_A[adapter].weight,
                          self.lora_B[adapter].weight, self.scaling[adapter], p)

    forward._soup_lora_fuse_patched = True
    forward._soup_lora_fuse_original = original
    Linear.forward = forward
