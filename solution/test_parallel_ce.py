"""Parallel vocabulary tiles preserve all head outputs and input gradient."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('shape', [(3, 17), (5, 4099), (9, 151936)])
@pytest.mark.parametrize('write_grad', [False, True])
def test_parallel_stats(dtype, shape, write_grad):
    from solution.parallel_ce import parallel_head_stats
    torch.manual_seed(21)
    rows, vocab = shape
    x = (torch.randn(rows, vocab, device='cuda') * 3).to(dtype)
    x[0].zero_()  # ties must select the first column
    labels = torch.randint(vocab, (rows,), device='cuda')
    labels[0] = 0
    labels[-1] = -100
    ref = x.float()
    lse = ref.logsumexp(-1)
    probs = (ref - lse[:, None]).exp()
    ent = lse - (probs * ref).sum(-1)
    loss_ref = lse - ref.gather(1, labels.clamp_min(0)[:, None]).squeeze(1)
    loss_ref[labels < 0] = 0
    hits_ref = (labels >= 0) & (ref.argmax(-1) == labels)
    loss = torch.empty(rows, device='cuda')
    entropy = torch.empty_like(loss)
    hit = torch.empty(rows, device='cuda', dtype=torch.int32)
    work = x.clone()
    parallel_head_stats(work, labels, loss, entropy, hit, write_grad)
    torch.testing.assert_close(loss, loss_ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(entropy, ent, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(hit.bool(), hits_ref)
    if write_grad:
        grad = probs
        labelled = labels >= 0
        grad[labelled, labels[labelled]] -= 1
        torch.testing.assert_close(work[labelled], grad[labelled].to(dtype), atol=1e-6, rtol=2e-4)
        torch.testing.assert_close(work[~labelled], x[~labelled], atol=0, rtol=0)
    else:
        torch.testing.assert_close(work, x, atol=0, rtol=0)


def test_parallel_head_gradient(monkeypatch):
    from solution.ce_loss import chunked_head_loss
    torch.manual_seed(44)
    h = torch.randn(2, 9, 64, device='cuda', requires_grad=True)
    w = torch.randn(4099, 64, device='cuda', dtype=torch.float16) * .1
    labels = torch.randint(4099, (2, 9), device='cuda')
    labels[:, :3] = -100
    outputs = []
    for enabled in ['0', '1']:
        monkeypatch.setenv('SOUP_X_PARALLEL_CE', enabled)
        loss, *rest = chunked_head_loss(h, w, labels, chunk=7)
        gradient, = torch.autograd.grad(loss, h)
        outputs.append((loss, rest, gradient))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(outputs[0][2], outputs[1][2], atol=3e-5, rtol=1e-3)
    for a, b in zip(outputs[0][1], outputs[1][1]):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
