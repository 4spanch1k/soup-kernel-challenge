"""NF4 weight dequantization in Triton, the largest op of the 4-bit base model that was not ours.

Every QLoRA linear layer turns its packed 4-bit weight into fp16 before the matmul, in forward
and again in backward (28 layers x 7 matrices x 2 per step). bitsandbytes does it with CUDA
kernels; here the same work is written in Triton. Backward calls
`bitsandbytes.functional.dequantize_4bit`, which is replaced; forward goes through the
`bitsandbytes::gemm_4bit` op, whose default implementation is "dequantize, then F.linear", so
`MatMul4Bit.forward` is replaced by the same two steps with our kernel (batch > 1 only; anything
else keeps bnb's own code).

Format (bitsandbytes 0.50, NF4 with double quantization), read from its source and checked
against its output in test_nf4_dequant.py:
- two weights per byte, the first one in the high nibble; the nibble indexes the 16-value NF4 table;
- one scale per `blocksize` (64) weights; with double quantization the scales are uint8 codes
  into a 256-entry table, themselves scaled per 256 scales, plus a scalar offset:
  scale = code2[q] * absmax2[block // 256] + offset;
- weight = nf4[nibble] * scale, computed in fp32 and rounded to the compute dtype (fp16).

A program handles `BLOCKS` quantization blocks as a [BLOCKS, blocksize/2] tile of bytes: loads are
coalesced, each scale is read once per block, and the two nibbles are interleaved back into
consecutive output elements before one contiguous store.
"""

import torch
import triton
import triton.language as tl

BLOCKS = 32  # quantization blocks per program (32 x 64 weights = 2048 outputs)


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax_q_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr,
                       out_ptr, n_blocks, BLOCK_SIZE: tl.constexpr, BLOCKS: tl.constexpr,
                       SECOND_BLOCK: tl.constexpr):
    block_ids = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)           # [BLOCKS]
    live = block_ids < n_blocks
    lane = tl.arange(0, BLOCK_SIZE // 2)                                   # byte within a block
    byte_idx = block_ids[:, None].to(tl.int64) * (BLOCK_SIZE // 2) + lane[None, :]
    packed = tl.load(packed_ptr + byte_idx, mask=live[:, None], other=0).to(tl.int32)

    quantized = tl.load(absmax_q_ptr + block_ids, mask=live, other=0).to(tl.int32)
    scale = (tl.load(code2_ptr + quantized) * tl.load(absmax2_ptr + block_ids // SECOND_BLOCK,
                                                      mask=live, other=0.0)
             + tl.load(offset_ptr))                                        # [BLOCKS] fp32

    high = tl.load(nf4_ptr + (packed >> 4)) * scale[:, None]
    low = tl.load(nf4_ptr + (packed & 15)) * scale[:, None]
    values = tl.interleave(high, low)                                      # [BLOCKS, BLOCK_SIZE]
    out_idx = (block_ids[:, None].to(tl.int64) * BLOCK_SIZE
               + tl.arange(0, BLOCK_SIZE)[None, :])
    tl.store(out_ptr + out_idx, values.to(out_ptr.dtype.element_ty), mask=live[:, None])


def nf4_dequantize(packed, quant_state):
    """Same result as bitsandbytes.functional.dequantize_4bit for a nested NF4 state."""
    state2 = quant_state.state2
    shape = quant_state.shape
    out = torch.empty(shape, dtype=quant_state.dtype, device=packed.device)
    n_blocks = out.numel() // quant_state.blocksize
    nf4_dequant_kernel[(triton.cdiv(n_blocks, BLOCKS),)](
        packed, quant_state.absmax, state2.code, state2.absmax, quant_state.offset.reshape(1),
        quant_state.code, out, n_blocks, BLOCK_SIZE=quant_state.blocksize, BLOCKS=BLOCKS,
        SECOND_BLOCK=state2.blocksize, num_warps=4)
    return out


def _supported(packed, quant_state, out, absmax):
    try:
        state2 = quant_state.state2
        return (out is None and absmax is None and packed.is_cuda and packed.dtype == torch.uint8
                and packed.shape[0] != 1 and quant_state.quant_type == "nf4"
                and quant_state.nested and quant_state.dtype == torch.float16
                and quant_state.blocksize in (64, 128, 256)
                and quant_state.absmax.dtype == torch.uint8 and state2.absmax.dtype == torch.float32
                and state2.code.dtype == torch.float32 and quant_state.code.numel() >= 16
                and quant_state.code.dtype == torch.float32 and len(quant_state.shape) == 2
                and (quant_state.shape[0] * quant_state.shape[1]) % quant_state.blocksize == 0
                and packed.numel() * 2 == quant_state.shape[0] * quant_state.shape[1])
    except Exception:  # noqa: BLE001 - any unexpected state shape: use bitsandbytes
        return False


def patch_bnb_dequantize():
    import bitsandbytes.functional as bnb_functional

    original = bnb_functional.dequantize_4bit
    if getattr(original, "_soup_nf4_patched", False):
        return

    def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=None,
                        quant_type="fp4"):
        if quant_state is not None and _supported(A, quant_state, out, absmax):
            return nf4_dequantize(A, quant_state)
        return original(A, quant_state=quant_state, absmax=absmax, out=out, blocksize=blocksize,
                        quant_type=quant_type)

    dequantize_4bit._soup_nf4_patched = True
    bnb_functional.dequantize_4bit = dequantize_4bit
    patch_bnb_matmul_forward()


def patch_bnb_matmul_forward():
    """MatMul4Bit.forward for the training case: our dequantization, then F.linear, exactly what
    bnb's default gemm_4bit does (dequantize to A's dtype, `F.linear(A, B_dq, bias)`)."""
    from bitsandbytes.autograd import _functions

    cls = _functions.MatMul4Bit
    if getattr(cls.forward, "_soup_nf4_patched", False):
        return
    original = cls.forward

    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        packed = B.reshape(-1, 1) if quant_state is not None else None
        if (quant_state is not None and out is None and A.is_cuda and A.numel() > 0
                and A.dtype == quant_state.dtype and _supported(packed, quant_state, None, None)):
            ctx.is_empty = False
            output = torch.nn.functional.linear(A, nf4_dequantize(packed, quant_state), bias)
            ctx.state = quant_state
            ctx.dtype_A, ctx.dtype_B = A.dtype, packed.dtype
            ctx.dtype_bias = None if bias is None else bias.dtype
            ctx.tensors = (None, packed) if any(ctx.needs_input_grad[:2]) else (None, None)
            return output
        return original(ctx, A, B, out, bias, quant_state)

    forward._soup_nf4_patched = True
    cls.forward = staticmethod(forward)
