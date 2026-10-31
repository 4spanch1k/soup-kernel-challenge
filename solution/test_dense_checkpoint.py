"""Opt-in attention checkpoint in the no-padding branch of the patched Qwen2 forward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _run(model, input_ids):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids,
                    use_cache=False)
    out.loss.backward()
    grad = model.model.layers[0].self_attn.q_proj.weight.grad.detach().clone()
    return out.loss.detach(), grad


def test_dense_checkpoint_matches_normal_forward_and_gradient(monkeypatch):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution.add_rmsnorm import patch_qwen2_decoder_layer
    from solution.layer_tail import patch_qwen2_layer_tail
    from solution.rmsnorm import patch_qwen2_rmsnorm

    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_layer()
    patch_qwen2_layer_tail()
    config = Qwen2Config(vocab_size=503, hidden_size=256, intermediate_size=512, num_hidden_layers=3,
                         num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True,
                         attention_dropout=0.0, use_cache=False)
    torch.manual_seed(0)
    reference = Qwen2ForCausalLM(config).cuda().train()
    checked = Qwen2ForCausalLM(config).cuda().train()
    checked.load_state_dict(reference.state_dict())
    input_ids = torch.randint(0, config.vocab_size, (3, 33), device="cuda")

    monkeypatch.delenv("SOUP_X_DENSE_CKPT", raising=False)
    ref_loss, ref_grad = _run(reference, input_ids)
    monkeypatch.setenv("SOUP_X_DENSE_CKPT", "1")
    loss, grad = _run(checked, input_ids)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-4)
    assert (grad - ref_grad).abs().max() <= 1e-2 * ref_grad.abs().max()
