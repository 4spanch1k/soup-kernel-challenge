"""Batched metric transport keeps Soup's values and uses one explicit host copy."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from solution.head_patch import HeadStep, _log_metrics


def _trainer(processes=1):
    calls = []

    def gather(value):
        calls.append(value)
        return value

    return SimpleNamespace(
        _metrics={"train": defaultdict(list)},
        _total_train_tokens=37,
        accelerator=SimpleNamespace(num_processes=processes, gather_for_metrics=gather),
        gather_calls=calls,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mask_kind", ["binary", "empty", "large_integer", "floating"])
def test_batched_metrics_equal_original(monkeypatch, device, mask_kind):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], device=device)
    if mask_kind == "empty":
        mask.zero_()
    elif mask_kind == "large_integer":
        # The transport must retain integers that fp32 would round.
        mask[0, 0] = 2**24 + 1
    elif mask_kind == "floating":
        mask = mask.float() * 0.5
    entropy = torch.tensor([[1.25, 2.5, 4.0], [3.5, 5.0, 6.0]], device=device)
    labelled = torch.tensor(0 if mask_kind == "empty" else 7, device=device)
    correct = torch.tensor(0 if mask_kind == "empty" else 3, device=device)
    step = HeadStep(entropy, None, correct, labelled)
    expected, actual = _trainer(), _trainer()

    monkeypatch.setenv("SOUP_X_METRICS_BATCHED", "0")
    _log_metrics(expected, {"attention_mask": mask}, step)
    monkeypatch.setenv("SOUP_X_METRICS_BATCHED", "1")
    _log_metrics(actual, {"attention_mask": mask}, step)

    assert dict(actual._metrics["train"]) == dict(expected._metrics["train"])
    assert actual._total_train_tokens == expected._total_train_tokens
    assert type(actual._total_train_tokens) is type(expected._total_train_tokens)
    assert len(actual.gather_calls) == len(expected.gather_calls) == 4


def test_one_host_copy_without_scalar_reads(monkeypatch):
    mask = torch.tensor([[1, 1, 0]])
    step = HeadStep(torch.tensor([[1.0, 3.0, 7.0]]), None, torch.tensor(1), torch.tensor(2))
    counts = defaultdict(int)
    original_item, original_bool = torch.Tensor.item, torch.Tensor.__bool__
    original_cpu = torch.Tensor.cpu

    def item(self, *args, **kwargs):
        counts["item"] += 1
        return original_item(self, *args, **kwargs)

    def boolean(self):
        counts["bool"] += 1
        return original_bool(self)

    def cpu(self, *args, **kwargs):
        counts["cpu"] += 1
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", item)
    monkeypatch.setattr(torch.Tensor, "__bool__", boolean)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setenv("SOUP_X_METRICS_BATCHED", "0")
    _log_metrics(_trainer(), {"attention_mask": mask}, step)
    assert counts == {"item": 3, "bool": 1}

    counts.clear()
    monkeypatch.setenv("SOUP_X_METRICS_BATCHED", "1")
    _log_metrics(_trainer(), {"attention_mask": mask}, step)
    assert counts == {"cpu": 1}


@pytest.mark.parametrize("processes", [2, None])
def test_distributed_or_unknown_process_count_keeps_original(monkeypatch, processes):
    import solution.head_patch as head_patch

    def forbidden(*args, **kwargs):
        pytest.fail("batched path used outside one-process accelerator")

    monkeypatch.setattr(head_patch, "_log_metrics_batched", forbidden)
    monkeypatch.setenv("SOUP_X_METRICS_BATCHED", "1")
    trainer = _trainer(processes)
    step = HeadStep(torch.ones(1, 2), None, torch.tensor(1), torch.tensor(2))
    _log_metrics(trainer, {"attention_mask": torch.ones(1, 2, dtype=torch.long)}, step)
    assert trainer._metrics["train"]["mean_token_accuracy"] == [0.5]
