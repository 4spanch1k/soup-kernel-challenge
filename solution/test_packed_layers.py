"""Packed Qwen2 layers against the original padded model forward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def make_models():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=503, hidden_size=256, intermediate_size=512,
                         num_hidden_layers=3, num_attention_heads=2,
                         num_key_value_heads=1, tie_word_embeddings=True,
                         attention_dropout=0.0, use_cache=False)
    torch.manual_seed(0)
    reference = Qwen2ForCausalLM(config).cuda().train()
    packed = Qwen2ForCausalLM(config).cuda().train()
    packed.load_state_dict(reference.state_dict())
    reference.get_input_embeddings().weight.requires_grad_(False)
    packed.get_input_embeddings().weight.requires_grad_(False)
    return reference, packed, config


def run_model(model, input_ids, attention_mask, labels):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                       use_cache=False)
    logits = output.logits.detach()
    output.loss.backward()
    grad = model.model.layers[0].self_attn.q_proj.weight.grad.detach().clone()
    return output.loss.detach(), logits, grad


@pytest.mark.parametrize("lengths", [(33, 20, 7), (2, 1, 1)])
def test_packed_forward_matches_original_loss_logits_and_gradient(lengths):
    from transformers.models.qwen2 import modeling_qwen2

    from solution.add_rmsnorm import patch_qwen2_decoder_layer
    from solution.layer_tail import patch_qwen2_layer_tail
    from solution.rmsnorm import patch_qwen2_rmsnorm

    reference, packed, config = make_models()
    input_ids = torch.randint(0, config.vocab_size, (3, 33), device="cuda")
    positions = torch.arange(33, device="cuda")[None, :]
    attention_mask = (positions < torch.tensor(lengths, device="cuda")[:, None]).long()
    labels = input_ids.masked_fill(~attention_mask.bool(), -100)

    current_forward = modeling_qwen2.Qwen2Model.forward
    original_forward = getattr(current_forward, "_soup_layer_tail_original", current_forward)
    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_layer()
    patch_qwen2_layer_tail()
    packed_forward = modeling_qwen2.Qwen2Model.forward
    input_dims = []
    hook = packed.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda _module, args: input_dims.append(args[0].dim()))

    try:
        modeling_qwen2.Qwen2Model.forward = original_forward
        ref_loss, ref_logits, ref_grad = run_model(reference, input_ids, attention_mask, labels)
        modeling_qwen2.Qwen2Model.forward = packed_forward
        loss, logits, grad = run_model(packed, input_ids, attention_mask, labels)
    finally:
        hook.remove()
        modeling_qwen2.Qwen2Model.forward = packed_forward

    assert input_dims and input_dims[0] == 2
    valid = attention_mask.bool()
    torch.testing.assert_close(logits[valid], ref_logits[valid], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(loss, ref_loss, rtol=2e-3, atol=2e-3)
    assert (grad - ref_grad).abs().max() <= 5e-2 * ref_grad.abs().max()


def test_unpadded_batch_uses_existing_padded_path():
    from transformers.models.qwen2 import modeling_qwen2

    from solution.add_rmsnorm import patch_qwen2_decoder_layer
    from solution.layer_tail import patch_qwen2_layer_tail
    from solution.rmsnorm import patch_qwen2_rmsnorm

    _, model, config = make_models()
    current_forward = modeling_qwen2.Qwen2Model.forward
    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_layer()
    patch_qwen2_layer_tail()

    input_ids = torch.randint(0, config.vocab_size, (3, 33), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    input_dims = []
    hook = model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda _module, args: input_dims.append(args[0].dim()))
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids,
                           use_cache=False)
        output.loss.backward()
    finally:
        hook.remove()
        modeling_qwen2.Qwen2Model.forward = getattr(
            current_forward, "_soup_layer_tail_original", current_forward)
        patch_qwen2_layer_tail()

    assert input_dims and input_dims[0] == 3


def _compare_with_original(attention_mask, lengths_valid=None):
    from transformers.models.qwen2 import modeling_qwen2

    from solution.add_rmsnorm import patch_qwen2_decoder_layer
    from solution.layer_tail import patch_qwen2_layer_tail
    from solution.rmsnorm import patch_qwen2_rmsnorm

    reference, fused, config = make_models()
    input_ids = torch.randint(0, config.vocab_size, attention_mask.shape, device="cuda")
    labels = input_ids.masked_fill(~attention_mask.bool(), -100)
    current_forward = modeling_qwen2.Qwen2Model.forward
    original_forward = getattr(current_forward, "_soup_layer_tail_original", current_forward)
    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_layer()
    patch_qwen2_layer_tail()
    fused_forward = modeling_qwen2.Qwen2Model.forward
    try:
        modeling_qwen2.Qwen2Model.forward = original_forward
        ref_loss, ref_logits, ref_grad = run_model(reference, input_ids, attention_mask, labels)
        modeling_qwen2.Qwen2Model.forward = fused_forward
        loss, logits, grad = run_model(fused, input_ids, attention_mask, labels)
    finally:
        modeling_qwen2.Qwen2Model.forward = fused_forward
    valid = attention_mask.bool()
    torch.testing.assert_close(logits[valid], ref_logits[valid], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(loss, ref_loss, rtol=2e-3, atol=2e-3)
    assert (grad - ref_grad).abs().max() <= 5e-2 * ref_grad.abs().max()


@pytest.mark.parametrize("mask_kind", ["ones", "none"])
def test_unpadded_batch_causal_attention_matches_original(mask_kind):
    if mask_kind == "ones":
        _compare_with_original(torch.ones(3, 33, dtype=torch.long, device="cuda"))
    else:
        from transformers.models.qwen2 import modeling_qwen2

        from solution.layer_tail import patch_qwen2_layer_tail

        reference, fused, config = make_models()
        input_ids = torch.randint(0, config.vocab_size, (3, 33), device="cuda")
        current_forward = modeling_qwen2.Qwen2Model.forward
        original_forward = getattr(current_forward, "_soup_layer_tail_original", current_forward)
        patch_qwen2_layer_tail()
        fused_forward = modeling_qwen2.Qwen2Model.forward
        try:
            modeling_qwen2.Qwen2Model.forward = original_forward
            ref_loss, ref_logits, _ = run_model(reference, input_ids, None, input_ids)
            modeling_qwen2.Qwen2Model.forward = fused_forward
            loss, logits, _ = run_model(fused, input_ids, None, input_ids)
        finally:
            modeling_qwen2.Qwen2Model.forward = fused_forward
        torch.testing.assert_close(logits, ref_logits, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(loss, ref_loss, rtol=2e-3, atol=2e-3)


def test_left_padded_batch_keeps_the_masked_attention_path():
    positions = torch.arange(33, device="cuda")[None, :]
    lengths = torch.tensor((33, 20, 7), device="cuda")[:, None]
    attention_mask = (positions >= 33 - lengths).long()  # padding on the left
    _compare_with_original(attention_mask)
