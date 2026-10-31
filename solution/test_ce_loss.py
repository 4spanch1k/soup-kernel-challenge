"""The chunked head + loss against plain PyTorch, and the Qwen2 patch end to end.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def make_case(batch=3, seq=37, width=64, vocab=5003, seed=0, hidden_dtype=torch.float32):
    torch.manual_seed(seed)
    hidden = torch.randn(batch, seq, width, device="cuda", dtype=hidden_dtype)
    weight = (torch.randn(vocab, width, device="cuda") * 0.05).half()
    lengths = torch.tensor([seq, seq - 9, 5], device="cuda")[:batch]
    mask = (torch.arange(seq, device="cuda")[None, :] < lengths[:, None]).long()
    prompt = torch.tensor([6, 3, 5], device="cuda")[:batch]
    labels = torch.randint(0, vocab, (batch, seq), device="cuda")
    # Half of the answer tokens are the model's own argmax, so "correct" is not always zero.
    best = (hidden.half() @ weight.t()).argmax(-1)
    take = torch.rand(batch, seq, device="cuda") < 0.5
    labels = torch.where(take, torch.roll(best, -1, dims=1), labels)
    answer = (torch.arange(seq, device="cuda")[None, :] >= prompt[:, None]) & mask.bool()
    labels = torch.where(answer, labels, torch.full_like(labels, -100))
    return hidden, weight, labels, mask


def reference(hidden, weight, labels, mask, den):
    h = hidden.detach().clone().requires_grad_(True)
    logits = (h.half() @ weight.t()).float()
    logp = F.log_softmax(logits, dim=-1)
    entropy = -(logp.exp() * logp).sum(-1)
    shifted = F.pad(labels, (0, 1), value=-100)[:, 1:]
    token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), shifted.reshape(-1),
                                 ignore_index=-100, reduction="none").view_as(shifted)
    n_lab = int((shifted >= 0).sum())
    loss = token_loss.sum() / (den if den is not None else max(n_lab, 1))
    loss.backward()
    correct = int(((logits.argmax(-1) == shifted) & (shifted >= 0)).sum())
    return loss.detach(), entropy.detach() * mask, token_loss.detach(), correct, n_lab, h.grad


@pytest.mark.parametrize("chunk", [7, 256])
@pytest.mark.parametrize("den", [None, 41.0])
@pytest.mark.parametrize("merge_variants", [False, True])
@pytest.mark.parametrize("hidden_dtype", [torch.float16, torch.float32])
def test_chunked_head_matches_pytorch(chunk, den, merge_variants, hidden_dtype, monkeypatch):
    from solution.ce_loss import chunked_head_loss

    if merge_variants:
        monkeypatch.setenv("SOUP_X_MERGE_VARIANTS", "1")
    else:
        monkeypatch.delenv("SOUP_X_MERGE_VARIANTS", raising=False)

    hidden, weight, labels, mask = make_case(hidden_dtype=hidden_dtype)
    ref_loss, ref_entropy, ref_token_loss, ref_correct, ref_lab, ref_grad = reference(
        hidden, weight, labels, mask, den)

    h = hidden.clone().requires_grad_(True)
    loss, entropy, token_loss, correct, labelled = chunked_head_loss(
        h, weight, labels, mask, None if den is None else torch.tensor(den, device="cuda"),
        chunk=chunk)
    (loss * 3.0).backward()  # backward must scale by the incoming gradient (loss scaling)

    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(token_loss, ref_token_loss, rtol=1e-3, atol=1e-3)
    assert int(labelled) == ref_lab
    assert int(correct) == ref_correct
    assert (h.grad - 3.0 * ref_grad).abs().max() <= 2e-2 * (3.0 * ref_grad).abs().max()


def test_padding_and_prompt_positions_get_no_gradient():
    from solution.ce_loss import chunked_head_loss

    hidden, weight, labels, mask = make_case()
    h = hidden.clone().requires_grad_(True)
    loss = chunked_head_loss(h, weight, labels, mask, None, chunk=16)[0]
    loss.backward()
    shifted = F.pad(labels, (0, 1), value=-100)[:, 1:]
    assert torch.all(h.grad[shifted < 0] == 0)
    assert torch.all(h.grad[shifted >= 0].abs().sum(-1) > 0)


def test_backward_with_fp16_loss_scale():
    """GradScaler multiplies the loss by 65536 before backward: the scale must reach the hidden
    gradient exactly once and the fp16 matmul must not overflow."""
    from solution.ce_loss import chunked_head_loss

    hidden, weight, labels, mask = make_case()
    den = 41.0
    _, _, _, _, _, ref_grad = reference(hidden, weight, labels, mask, den)
    h = hidden.clone().requires_grad_(True)
    loss = chunked_head_loss(h, weight, labels, mask, torch.tensor(den, device="cuda"), chunk=16)[0]
    (loss * 65536.0).backward()
    assert torch.isfinite(h.grad).all()
    expected = 65536.0 * ref_grad
    assert (h.grad - expected).abs().max() <= 2e-2 * expected.abs().max()


@pytest.mark.parametrize("denominator", [0.0, 0])
def test_zero_denominator_gives_zero_not_nan(denominator):
    from solution.ce_loss import chunked_head_loss

    hidden, weight, labels, mask = make_case()
    h = hidden.clone().requires_grad_(True)
    loss = chunked_head_loss(h, weight, torch.full_like(labels, -100), mask, denominator)[0]
    loss.backward()
    assert float(loss) == 0.0 and torch.isfinite(h.grad).all()


def test_one_token_rows_have_no_target():
    from solution.ce_loss import chunked_head_loss

    hidden, weight, labels, _ = make_case(batch=2, seq=1)
    h = hidden.clone().requires_grad_(True)
    one = torch.ones(2, 1, dtype=torch.long, device="cuda")
    loss, entropy, token_loss, correct, labelled = chunked_head_loss(h, weight, labels, one, None)
    loss.backward()
    assert float(loss) == 0.0 and int(labelled) == 0 and torch.all(entropy > 0)


def test_no_labels_in_batch():
    from solution.ce_loss import chunked_head_loss

    hidden, weight, labels, mask = make_case()
    h = hidden.clone().requires_grad_(True)
    loss, entropy, token_loss, correct, labelled = chunked_head_loss(
        h, weight, torch.full_like(labels, -100), mask, None)
    loss.backward()
    assert float(loss) == 0.0 and int(labelled) == 0 and int(correct) == 0
    assert torch.all(h.grad == 0)
    assert torch.all(entropy[mask.bool()] > 0)  # entropy is still logged for prompt tokens


def test_qwen2_patch_matches_original_forward():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution import head_patch

    head_patch.patch_forward()
    config = Qwen2Config(vocab_size=1003, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         tie_word_embeddings=True)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).cuda().train()
    model.get_input_embeddings().weight.requires_grad_(False)  # tied with lm_head, frozen
    hidden, _, labels, mask = make_case(vocab=1003)
    ids = torch.randint(0, 1003, labels.shape, device="cuda")
    num_items = torch.tensor(float((F.pad(labels, (0, 1), value=-100)[:, 1:] >= 0).sum()),
                             device="cuda")
    param = model.model.layers[0].self_attn.q_proj.weight

    def run(fast):
        model.zero_grad()
        head_patch._FAST.armed = fast
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=ids, attention_mask=mask, labels=labels,
                            num_items_in_batch=num_items)
        finally:
            head_patch._FAST.armed = False
        out.loss.backward()
        return out, param.grad.clone()

    base_out, base_grad = run(False)
    fast_out, fast_grad = run(True)
    assert base_out.logits is not None and fast_out.logits is None
    torch.testing.assert_close(fast_out.loss, base_out.loss, rtol=2e-3, atol=2e-3)
    assert (fast_grad - base_grad).abs().max() <= 5e-2 * base_grad.abs().max()
    step = head_patch._FAST.step
    assert step is not None and step.token_loss.shape == labels.shape
