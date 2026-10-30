"""Tied embedding compaction against the original Qwen2 fp32 path."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def make_model(vocab_size=503, hidden_size=64, tie=True):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=vocab_size, hidden_size=hidden_size,
                         intermediate_size=hidden_size * 2, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1,
                         tie_word_embeddings=tie, use_cache=False)
    return Qwen2ForCausalLM(config).cuda().train()


def make_fp16_representable(model):
    with torch.no_grad():
        weight = model.model.embed_tokens.weight
        weight.copy_(weight.half().float())
        weight.requires_grad_(False)
        if model.lm_head.weight is not weight:
            model.lm_head.weight.requires_grad_(False)


def run(model, input_ids):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    loss.backward()
    grad = model.model.layers[0].self_attn.q_proj.weight.grad.detach().clone()
    return loss.detach(), grad


def test_compaction_preserves_loss_gradient_and_tying():
    from solution.embed16 import compact_tied_embedding
    from solution.head_patch import _head_weight

    torch.manual_seed(0)
    reference = make_model()
    compact = make_model()
    compact.load_state_dict(reference.state_dict())
    make_fp16_representable(reference)
    make_fp16_representable(compact)
    input_ids = torch.randint(0, reference.config.vocab_size, (2, 17), device="cuda")

    ref_loss, ref_grad = run(reference, input_ids)
    assert compact.model.embed_tokens.weight.dtype == torch.float32
    assert compact_tied_embedding(compact)
    assert compact.model.embed_tokens.weight is compact.lm_head.weight
    assert compact.model.embed_tokens.weight.dtype == torch.float16
    assert not compact.model.embed_tokens.weight.requires_grad
    assert compact.get_input_embeddings()(input_ids).dtype == torch.float32
    compact.enable_input_require_grads()
    hooked_output = compact.get_input_embeddings()(input_ids)
    assert hooked_output.dtype == torch.float32 and hooked_output.requires_grad
    assert _head_weight(compact.lm_head, torch.float16) is compact.lm_head.weight

    loss, grad = run(compact, input_ids)
    torch.testing.assert_close(loss, ref_loss, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grad, ref_grad, rtol=5e-2, atol=5e-2)


def test_compaction_releases_fp32_embedding_memory():
    from solution.embed16 import compact_tied_embedding

    model = make_model(vocab_size=8192, hidden_size=256)
    make_fp16_representable(model)
    numel = model.model.embed_tokens.weight.numel()
    before = torch.cuda.memory_allocated()

    assert compact_tied_embedding(model)
    torch.cuda.synchronize()
    released = before - torch.cuda.memory_allocated()
    expected = numel * (4 - 2)
    assert abs(released - expected) <= max(expected * 0.1, 1 << 20)


def test_guards_leave_unrepresentable_and_untied_weights_alone():
    from solution.embed16 import compact_tied_embedding

    tied = make_model()
    make_fp16_representable(tied)
    with torch.no_grad():
        tied.model.embed_tokens.weight[0, 0] = 0.1
    original = tied.model.embed_tokens.weight
    assert not compact_tied_embedding(tied)
    assert tied.model.embed_tokens.weight is original
    assert tied.lm_head.weight is original
    assert original.dtype == torch.float32

    untied = make_model(tie=False)
    make_fp16_representable(untied)
    embedding_weight, head_weight = untied.model.embed_tokens.weight, untied.lm_head.weight
    assert not compact_tied_embedding(untied)
    assert untied.model.embed_tokens.weight is embedding_weight
    assert untied.lm_head.weight is head_weight
    assert embedding_weight.dtype == head_weight.dtype == torch.float32


def test_compaction_accepts_separate_parameters_on_the_same_storage():
    from solution.embed16 import compact_tied_embedding

    model = make_model()
    make_fp16_representable(model)
    model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.detach(),
                                               requires_grad=False)
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    assert compact_tied_embedding(model)
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_peft_lifecycle_backward_save_and_eval_without_autocast(tmp_path):
    """The order Soup uses: PEFT wraps the model first, compaction happens at the first training
    forward, then backward with LoRA dropout, the adapter save, and a plain same-process eval."""
    import os

    from peft import LoraConfig, get_peft_model

    from solution.embed16 import compact_tied_embedding

    base = make_model()
    make_fp16_representable(base)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.05,
                                            target_modules=["q_proj", "v_proj"],
                                            task_type="CAUSAL_LM"))
    assert compact_tied_embedding(model.get_base_model())
    embedding = model.get_input_embeddings()(torch.ones((1, 8), dtype=torch.long, device="cuda"))
    assert embedding.dtype == torch.float32

    ids = torch.randint(0, base.config.vocab_size, (2, 16), device="cuda")
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    assert torch.isfinite(loss)

    model.save_pretrained(str(tmp_path))
    assert os.path.exists(os.path.join(str(tmp_path), "adapter_model.safetensors"))

    model.eval()
    with torch.no_grad():  # no autocast: the head must still accept the fp32 hidden states
        logits = model(input_ids=ids).logits
    assert torch.isfinite(logits.float()).all()
