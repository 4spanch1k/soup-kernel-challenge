"""Compare our serial and parallel vocabulary statistics, outside scored runs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton
from solution.ce_loss import head_stats_kernel
from solution.parallel_ce import parallel_head_stats
for rows in (64, 256):
    vocab = 151936
    logits = torch.randn(rows, vocab, device='cuda', dtype=torch.float16)
    target = torch.randint(vocab, (rows,), device='cuda')
    loss = torch.empty(rows, device='cuda')
    ent = torch.empty_like(loss)
    hit = torch.empty(rows, device='cuda', dtype=torch.int32)
    for grad in (False, True):
        def old():
            head_stats_kernel[(rows,)](logits, target, loss, ent, hit, vocab, vocab,
                                      BLOCK=4096, WRITE_GRAD=grad, num_warps=8)
        def new():
            parallel_head_stats(logits, target, loss, ent, hit, grad)
        # Restore logits outside timed sections. Their values do not change launch geometry.
        for name, fn in [('serial', old), ('parallel', new)]:
            print(rows, grad, name, triton.testing.do_bench(fn, warmup=100, rep=300), 'ms', flush=True)
