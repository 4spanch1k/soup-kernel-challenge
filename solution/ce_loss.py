"""Chunked output head + cross-entropy in Triton: the [tokens, 151936] logits never exist at once.

The base run builds the full logits (fp16), upcasts them to fp32, runs log_softmax / cross-entropy,
and TRL then makes more copies for entropy and accuracy. This file does the head and everything
Soup reads from the logits in token chunks of `CHUNK` rows:

- one Triton kernel per chunk reads each row of logits twice (online softmax, then the gradient)
  and writes, per token: cross-entropy, entropy, "argmax == label", and the gradient
  `softmax - onehot` in place of the logits;
- the gradient on the hidden states (`grad_logits @ W`) is taken right away, chunk by chunk, so
  backward only scales and scatters it (the loss scale of fp16 training arrives in backward);
- rows are reordered so labelled tokens come first: padding is never computed, and the backward
  matmul runs for answer tokens only (prompt tokens have no label and no gradient, but they still
  need their entropy, which Soup logs over every non-padding token).

Ideas: online softmax (Milakov & Gimelshein, 2018); computing the head and loss in chunks (as in
Liger's fused linear cross-entropy and Apple's Cut Cross-Entropy). The code is written here.
"""

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

IGNORE_INDEX = -100
CHUNK = 256      # token rows per chunk: a [256, 151936] fp16 buffer is about 74 MiB
BLOCK_V = 4096   # vocabulary columns per loop step inside the kernel


@triton.jit
def head_stats_kernel(logits_ptr, target_ptr, loss_ptr, entropy_ptr, hit_ptr, row_stride, vocab,
                      BLOCK: tl.constexpr, WRITE_GRAD: tl.constexpr):
    """One program per token row. target < 0 means "no label" (loss 0, no hit)."""
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * row_stride
    target = tl.load(target_ptr + row)
    lane = tl.arange(0, BLOCK)

    # Reductions of constants give 0-d tensors of the loop-carried type.
    run_max = tl.max(tl.full([BLOCK], float("-inf"), tl.float32), 0)
    run_sum = tl.sum(tl.zeros([BLOCK], tl.float32), 0)
    run_wsum = tl.sum(tl.zeros([BLOCK], tl.float32), 0)
    best = run_max
    best_idx = tl.sum(tl.zeros([BLOCK], tl.int32), 0)
    for start in range(0, vocab, BLOCK):
        cols = start + lane
        inside = cols < vocab
        x = tl.load(base + cols, mask=inside, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x, 0)
        new_max = tl.maximum(run_max, block_max)
        shrink = tl.exp(run_max - new_max)
        e = tl.exp(x - new_max)
        run_sum = run_sum * shrink + tl.sum(e, 0)
        safe_x = tl.where(inside, x, 0.0)  # padded lanes are -inf: keep 0 * -inf out of the sum
        run_wsum = run_wsum * shrink + tl.sum(tl.where(inside, e, 0.0) * safe_x, 0)
        better = block_max > best
        best_idx = tl.where(better, start + tl.argmax(x, 0), best_idx)
        best = tl.maximum(best, block_max)
        run_max = new_max

    lse = run_max + tl.log(run_sum)
    has_target = target >= 0
    at_target = tl.load(base + tl.maximum(target, 0)).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(has_target, lse - at_target, 0.0))
    tl.store(entropy_ptr + row, lse - run_wsum / run_sum)  # H = lse - sum(p * x)
    tl.store(hit_ptr + row, tl.where(has_target & (best_idx == target), 1, 0))

    if WRITE_GRAD:
        grad_end = tl.where(has_target, vocab, 0)  # prompt-only rows share a chunk: skip them
        for start in range(0, grad_end, BLOCK):
            cols = start + lane
            inside = cols < vocab
            x = tl.load(base + cols, mask=inside, other=float("-inf")).to(tl.float32)
            grad = tl.exp(x - lse) - tl.where(cols == target, 1.0, 0.0)
            tl.store(base + cols, grad.to(logits_ptr.dtype.element_ty), mask=inside)


def _forward(ctx, hidden, weight, labels, attention_mask, denominator, chunk):
    batch, seq, width = hidden.shape
    vocab = weight.shape[0]
    device = hidden.device
    need_grad = ctx.needs_input_grad[0]
    merge_variants = os.environ.get("SOUP_X_MERGE_VARIANTS") == "1"

    flat = hidden.reshape(batch * seq, width)
    # Position t is scored against token t + 1; the last position has no target.
    target = F.pad(labels, (0, 1), value=IGNORE_INDEX)[:, 1:].reshape(-1)
    labelled = target >= 0
    valid = labelled if attention_mask is None else (labelled | attention_mask.reshape(-1).bool())
    lab_idx = labelled.nonzero().squeeze(1)
    unl_idx = (valid & ~labelled).nonzero().squeeze(1)
    perm = torch.cat([lab_idx, unl_idx])  # labelled rows first
    n_lab, n = lab_idx.numel(), perm.numel()

    token_entropy = torch.zeros(batch * seq, dtype=torch.float32, device=device)
    token_loss = torch.zeros(batch * seq, dtype=torch.float32, device=device)
    total = torch.zeros((), dtype=torch.float32, device=device)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    grad_rows = None
    if n:
        rows = flat.index_select(0, perm)
        row_target = target.index_select(0, perm).to(torch.int32)
        loss_rows = torch.empty(n, dtype=torch.float32, device=device)
        entropy_rows = torch.empty(n, dtype=torch.float32, device=device)
        hit_rows = torch.empty(n, dtype=torch.int32, device=device)
        if need_grad and n_lab:
            grad_rows = torch.empty(n_lab, width, dtype=torch.float32, device=device)
        buffer = torch.empty(min(chunk, n), vocab, dtype=weight.dtype, device=device)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            logits = buffer[:stop - start]
            torch.mm(rows[start:stop].to(weight.dtype), weight.t(), out=logits)
            lab_in_chunk = max(0, min(stop, n_lab) - start)
            # Prompt-only rows have no target, so the kernel skips their gradient loop.
            head_stats_kernel[(stop - start,)](
                logits, row_target[start:stop], loss_rows[start:stop], entropy_rows[start:stop],
                hit_rows[start:stop], logits.stride(0), vocab, BLOCK=BLOCK_V,
                WRITE_GRAD=(True if merge_variants else bool(need_grad and lab_in_chunk)),
                num_warps=8)
            if need_grad and lab_in_chunk:
                grad_rows[start:start + lab_in_chunk] = torch.mm(logits[:lab_in_chunk], weight)
        total = loss_rows.sum()
        correct = hit_rows.sum(dtype=torch.int64)
        token_entropy[perm] = entropy_rows
        token_loss[perm] = loss_rows

    if denominator is None:
        den = torch.full((), float(max(n_lab, 1)), dtype=torch.float32, device=device)
    else:  # a Python number or a device scalar; never read back to the host
        den = torch.as_tensor(denominator, dtype=torch.float32, device=device).clamp_min(1.0)
    ctx.den, ctx.hidden_shape, ctx.hidden_dtype = den, hidden.shape, hidden.dtype
    ctx.save_for_backward(grad_rows if grad_rows is not None else lab_idx.new_empty(0), lab_idx)
    ctx.has_grad_rows = grad_rows is not None
    n_labelled = torch.tensor(n_lab, dtype=torch.int64, device=device)
    return (total / den, token_entropy.view(batch, seq), token_loss.view(batch, seq), correct,
            n_labelled)


class ChunkedHeadLoss(torch.autograd.Function):
    """loss = sum of token cross-entropies / denominator, plus the per-token numbers Soup logs.

    Outputs: loss, token_entropy [B, T], token_loss [B, T] (position t holds the loss of token
    t + 1, 0 where there is no label), correct (answer tokens predicted right), labelled count.
    """

    @staticmethod
    def forward(ctx, hidden, weight, labels, attention_mask, denominator, chunk):
        with torch.autocast("cuda", enabled=False):
            outputs = _forward(ctx, hidden, weight, labels, attention_mask, denominator, chunk)
        ctx.mark_non_differentiable(*outputs[1:])
        return outputs

    @staticmethod
    def backward(ctx, grad_loss, *_):
        grad_rows, lab_idx = ctx.saved_tensors
        batch, seq, width = ctx.hidden_shape
        grad_hidden = torch.zeros(batch * seq, width, dtype=ctx.hidden_dtype,
                                  device=lab_idx.device)
        if ctx.has_grad_rows:
            scale = grad_loss.float() / ctx.den  # the loss scale of fp16 training arrives here
            grad_hidden[lab_idx] = (grad_rows * scale).to(ctx.hidden_dtype)
        return grad_hidden.view(batch, seq, width), None, None, None, None, None


def chunked_head_loss(hidden, weight, labels, attention_mask=None, denominator=None, chunk=CHUNK):
    """`hidden` [B, T, H] (any float dtype), `weight` [V, H] already in the matmul dtype (fp16),
    `labels` [B, T] with -100 on unsupervised positions, `attention_mask` [B, T] or None.
    `denominator` is Trainer's num_items_in_batch; None means the number of labelled tokens."""
    return ChunkedHeadLoss.apply(hidden, weight, labels, attention_mask, denominator, chunk)
