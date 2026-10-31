"""Real PEFT forwarding must reach the packed path; output requests retain fallback."""

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _config():
    from transformers import Qwen2Config

    return Qwen2Config(vocab_size=503, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, tie_word_embeddings=True,
                      attention_dropout=0.0, use_cache=False)


@pytest.mark.parametrize("extra", [
    {"output_hidden_states": None, "output_attentions": None, "return_dict": None},
    {"output_hidden_states": False, "output_attentions": False, "return_dict": True},
])
def test_real_peft_uses_packed_path_and_matches_outputs_and_gradients(monkeypatch, extra):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen2ForCausalLM
    from transformers.models.qwen2 import modeling_qwen2

    from solution import head_patch
    from solution.layer_tail import patch_qwen2_layer_tail
    from solution.rmsnorm import patch_qwen2_rmsnorm

    torch.manual_seed(71)
    model = get_peft_model(
        Qwen2ForCausalLM(_config()).half().float(),
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                   target_modules=["q_proj", "v_proj"], lora_dropout=0.0),
    ).cuda().train()
    # Nonzero B exercises both adapter gradients. Dropout is zero for this
    # equivalence test; the task's unchanged dropout is checked by full harness.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.02)
    ids = torch.randint(0, 503, (3, 17), device="cuda")
    mask = (torch.arange(17, device="cuda")[None, :]
            < torch.tensor([17, 11, 5], device="cuda")[:, None]).long()
    labels = ids.masked_fill(mask == 0, -100)
    cls = modeling_qwen2.Qwen2Model
    current = cls.forward
    original = getattr(current, "_soup_layer_tail_original", current)
    monkeypatch.setattr(cls, "forward", original)
    patch_qwen2_layer_tail()
    packed_forward = cls.forward
    patch_qwen2_rmsnorm()
    head_patch.patch_forward()
    monkeypatch.setattr(head_patch._FAST, "armed", True)

    core = model.get_base_model()
    input_dimensions = []
    hook = core.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda _module, args: input_dimensions.append(args[0].dim()))

    def run(forward):
        monkeypatch.setattr(cls, "forward", forward)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(input_ids=ids, attention_mask=mask, labels=labels,
                           use_cache=False, **extra)
        output.loss.backward()
        grads = {name: parameter.grad.detach().clone()
                 for name, parameter in model.named_parameters() if "lora_" in name}
        return output.loss.detach(), grads

    try:
        expected, expected_grads = run(original)
        actual, actual_grads = run(packed_forward)
    finally:
        hook.remove()
    assert input_dimensions == [3, 2]
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    assert actual_grads.keys() == expected_grads.keys()
    for name, value in actual_grads.items():
        reference = expected_grads[name]
        relative_rmse = (value.float() - reference.float()).square().mean().sqrt()
        relative_rmse /= reference.float().square().mean().sqrt().clamp_min(1e-8)
        assert relative_rmse < 0.05, name


@pytest.mark.parametrize("options", [
    {"output_hidden_states": True}, {"output_attentions": True},
    {"return_dict": False}, {"unknown_option": None}, {"cache_position": None},
])
def test_nondefault_or_unknown_kwargs_reach_original_unchanged(monkeypatch, options):
    from transformers import Qwen2Model
    from transformers.models.qwen2 import modeling_qwen2

    from solution.layer_tail import patch_qwen2_layer_tail

    calls = []
    sentinel = object()

    def original(self, *args, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(modeling_qwen2.Qwen2Model, "forward", original)
    patch_qwen2_layer_tail()
    model = Qwen2Model(_config()).cuda().train()
    ids = torch.randint(0, 503, (2, 7), device="cuda")
    assert model(input_ids=ids, use_cache=False, **options) is sentinel
    assert len(calls) == 1
    assert all(calls[0][key] is value for key, value in options.items())


def test_fallback_trace_is_once_per_model_and_contains_no_tensor_values(monkeypatch, capsys):
    from transformers import Qwen2Model
    from transformers.models.qwen2 import modeling_qwen2

    from solution.layer_tail import patch_qwen2_layer_tail

    monkeypatch.setenv("SOUP_X_TRACE_DISPATCH", "1")
    monkeypatch.setattr(modeling_qwen2.Qwen2Model, "forward", lambda *_args, **_kwargs: None)
    patch_qwen2_layer_tail()
    model = Qwen2Model(_config()).cuda().train()
    ids = torch.randint(0, 503, (2, 7), device="cuda")
    for _ in range(2):
        model(input_ids=ids, use_cache=False, unknown_option=ids)
    text = capsys.readouterr().out
    assert text.count("SOUP_LAYER_TAIL_FALLBACK") == 1
    assert '"unknown_option": "Tensor"' in text
    assert '"reason": "model_guard"' in text
    assert "tensor(" not in text
