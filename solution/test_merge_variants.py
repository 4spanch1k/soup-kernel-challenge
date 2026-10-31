"""Merged Triton specializations preserve RoPE, CE and residual results."""

import importlib
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_ce_merge_variants_matches_default(monkeypatch):
    from solution.ce_loss import chunked_head_loss

    torch.manual_seed(23)
    hidden = torch.randn(2, 7, 17, device="cuda", dtype=torch.float16)
    weight = torch.randn(101, 17, device="cuda", dtype=torch.float16)
    labels = torch.randint(0, 101, (2, 7), device="cuda")
    labels[:, 0] = -100
    mask = torch.ones_like(labels)

    def run():
        x = hidden.detach().clone().requires_grad_(True)
        result = chunked_head_loss(x, weight, labels, mask, chunk=3)
        result[0].backward()
        return tuple(value.detach() for value in result), x.grad

    monkeypatch.delenv("SOUP_X_MERGE_VARIANTS", raising=False)
    normal = run()
    monkeypatch.setenv("SOUP_X_MERGE_VARIANTS", "1")
    merged = run()
    for normal_value, merged_value in zip(normal[0], merged[0]):
        assert torch.equal(normal_value, merged_value)
    assert torch.equal(normal[1], merged[1])


def test_residual_merge_variants_matches_default(monkeypatch):
    import solution.add_rmsnorm as add_module
    original_mode = os.environ.get("SOUP_X_MERGE_VARIANTS")

    def run(merge, use_sum_grad):
        if merge:
            monkeypatch.setenv("SOUP_X_MERGE_VARIANTS", "1")
        else:
            monkeypatch.delenv("SOUP_X_MERGE_VARIANTS", raising=False)
        importlib.reload(add_module)
        torch.manual_seed(29)
        residual = torch.randn(3, 7, 1001, device="cuda", requires_grad=True)
        delta = torch.randn(3, 7, 1001, device="cuda", dtype=torch.float16, requires_grad=True)
        weight = torch.rand(1001, device="cuda")
        y, total = add_module.add_rmsnorm(residual, delta, weight, 1e-6, torch.float16)
        if use_sum_grad:
            torch.autograd.backward((y, total), (torch.randn_like(y), torch.randn_like(total)))
        else:
            y.sum().backward()
        return y.detach(), total.detach(), residual.grad, delta.grad

    normal = run(False, True)
    merged_with_sum = run(True, True)
    for left, right in zip(normal, merged_with_sum):
        assert torch.equal(left, right)

    normal_without_sum = run(False, False)
    merged_without_sum = run(True, False)
    for left, right in zip(normal_without_sum, merged_without_sum):
        assert torch.equal(left, right)
    if original_mode is None:
        monkeypatch.delenv("SOUP_X_MERGE_VARIANTS", raising=False)
    else:
        monkeypatch.setenv("SOUP_X_MERGE_VARIANTS", original_mode)
    importlib.reload(add_module)


def test_rope_merge_variants_matches_default(monkeypatch):
    import solution.rope as rope_module
    original_mode = os.environ.get("SOUP_X_MERGE_VARIANTS")

    torch.manual_seed(37)
    source = torch.randn(2, 3, 7, 16, device="cuda", dtype=torch.float16)
    cosine = torch.randn(1, 7, 16, device="cuda")
    sine = torch.randn(1, 7, 16, device="cuda")
    grad = torch.randn_like(source)

    def run(merge):
        if merge:
            monkeypatch.setenv("SOUP_X_MERGE_VARIANTS", "1")
        else:
            monkeypatch.delenv("SOUP_X_MERGE_VARIANTS", raising=False)
        importlib.reload(rope_module)
        x = source.detach().clone().requires_grad_(True)
        output = rope_module.RotaryFunction.apply(x, cosine, sine, torch.float16)
        (output * grad).sum().backward()
        return output.detach(), x.grad

    normal = run(False)
    merged = run(True)
    for left, right in zip(normal, merged):
        assert torch.equal(left, right)
    if original_mode is None:
        monkeypatch.delenv("SOUP_X_MERGE_VARIANTS", raising=False)
    else:
        monkeypatch.setenv("SOUP_X_MERGE_VARIANTS", original_mode)
    importlib.reload(rope_module)
