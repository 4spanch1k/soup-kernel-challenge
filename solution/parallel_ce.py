"""Own parallel vocabulary statistics for the chunked head.

Idea: associative max/sum reduction, in three launches instead of one long serial
vocabulary loop. Inspired by the parallel reduction described in competitors'
TEAM_INTEGRATION.md; code here is independently written. Disabled by default.
"""
import torch
import triton
import triton.language as tl

TILE = 4096


@triton.jit
def vocab_partials_kernel(logits, partial_max, partial_sum, partial_weighted, partial_arg,
                          stride, vocab, parts: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    valid = col < vocab
    x = tl.load(logits + row.to(tl.int64) * stride + col, valid, float('-inf')).to(tl.float32)
    maximum = tl.max(x, 0)
    exp_x = tl.exp(x - maximum)
    sum_x = tl.sum(exp_x, 0)
    weighted = tl.sum(exp_x * tl.where(valid, x, 0.), 0)
    best = tl.min(tl.where((x == maximum) & valid, col, 2147483647), 0)
    out = row * parts + part
    tl.store(partial_max + out, maximum)
    tl.store(partial_sum + out, sum_x)
    tl.store(partial_weighted + out, weighted)
    tl.store(partial_arg + out, best)


@triton.jit
def vocab_finish_kernel(logits, target, pm, ps, pw, pa, loss, entropy, hit, row_lse,
                        stride, parts: tl.constexpr, REDUCE: tl.constexpr):
    row = tl.program_id(0)
    p = tl.arange(0, REDUCE)
    live = p < parts
    offset = row * parts + p
    maxima = tl.load(pm + offset, live, float('-inf'))
    maximum = tl.max(maxima, 0)
    factors = tl.exp(maxima - maximum)
    total = tl.sum(tl.load(ps + offset, live, 0.) * factors, 0)
    weighted = tl.sum(tl.load(pw + offset, live, 0.) * factors, 0)
    best_col = tl.min(tl.where(maxima == maximum,
                     tl.load(pa + offset, live, 2147483647), 2147483647), 0)
    lse = maximum + tl.log(total)
    label = tl.load(target + row)
    label_logit = tl.load(logits + row.to(tl.int64) * stride + tl.maximum(label, 0)).to(tl.float32)
    tl.store(loss + row, tl.where(label >= 0, lse - label_logit, 0.))
    tl.store(entropy + row, lse - weighted / total)
    tl.store(hit + row, ((label >= 0) & (label == best_col)).to(tl.int32))
    tl.store(row_lse + row, lse)


@triton.jit
def vocab_gradient_kernel(logits, target, row_lse, stride, vocab, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    label = tl.load(target + row)
    live = (col < vocab) & (label >= 0)
    offset = row.to(tl.int64) * stride + col
    x = tl.load(logits + offset, live, float('-inf')).to(tl.float32)
    lse = tl.load(row_lse + row)
    grad = tl.exp(x - lse) - (col == label).to(tl.float32)
    tl.store(logits + offset, grad.to(logits.dtype.element_ty), live)


def parallel_head_stats(logits, target, loss, entropy, hit, write_grad):
    rows, vocab = logits.shape
    parts = triton.cdiv(vocab, TILE)
    stats = [torch.empty((rows, parts), device=logits.device, dtype=torch.float32) for _ in range(3)]
    args = torch.empty((rows, parts), device=logits.device, dtype=torch.int32)
    lse = torch.empty(rows, device=logits.device, dtype=torch.float32)
    vocab_partials_kernel[(rows, parts)](logits, *stats, args, logits.stride(0), vocab,
                                        parts=parts, BLOCK=TILE, num_warps=4)
    vocab_finish_kernel[(rows,)](logits, target, *stats, args, loss, entropy, hit, lse,
                                 logits.stride(0), parts=parts,
                                 REDUCE=triton.next_power_of_2(parts), num_warps=4)
    if write_grad:
        vocab_gradient_kernel[(rows, parts)](logits, target, lse, logits.stride(0), vocab,
                                             BLOCK=TILE, num_warps=4)
