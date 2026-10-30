"""LoRA fusion against PEFT and its regenerated dropout mask."""

import copy
import gc

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _peft():
    try:
        from peft import LoraConfig, get_peft_model
        from peft.tuners.lora.layer import Linear
    except Exception as exc:  # optional dependency can fail during torchao imports
        pytest.skip(f"PEFT unavailable (possibly an incompatible torchao install): {exc}")
    return LoraConfig, get_peft_model, Linear


def _tiny_lora_model(dropout):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    LoraConfig, get_peft_model, _ = _peft()
    config = Qwen2Config(vocab_size=101, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         attention_dropout=0.0, use_cache=False)
    base = Qwen2ForCausalLM(config)
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                      lora_dropout=dropout, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(base, lora).cuda().train()


def _run_model(model, input_ids):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    loss.backward()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()
             if "lora_A" in name or "lora_B" in name}
    return loss.detach(), grads


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_and_backward_match_pytorch_on_odd_shapes(dtype):
    from solution.lora_fuse import lora_fused

    torch.manual_seed(4)
    x = torch.randn(2, 3, 13, device="cuda", dtype=dtype, requires_grad=True)
    base = torch.randn(2, 3, 7, device="cuda", dtype=dtype, requires_grad=True)
    a = torch.randn(3, 13, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(7, 3, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn_like(base)
    ref = (base + F.linear(F.linear(x, a), b) * 1.5).to(base.dtype)
    ref.backward(grad)
    ref_out = ref.detach().clone()
    ref_grads = [item.grad.detach().clone() for item in (x, base, a, b)]

    x.grad = base.grad = a.grad = b.grad = None
    fused = lora_fused(x, base, a, b, 1.5, 0.0)
    fused.backward(grad)
    tolerance = 2e-2 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(fused, ref_out, rtol=tolerance, atol=tolerance)
    for actual, expected in zip((x.grad, base.grad, a.grad, b.grad), ref_grads):
        torch.testing.assert_close(actual, expected, rtol=5 * tolerance, atol=5 * tolerance)


def test_peft_forward_and_adapter_gradients_match_without_dropout(monkeypatch):
    from solution import lora_fuse

    monkeypatch.setenv("SOUP_LORA_FUSE", "1")
    _, _, Linear = _peft()
    original = getattr(Linear.forward, "_soup_lora_fuse_original", Linear.forward)
    Linear.forward = original
    reference = _tiny_lora_model(0.0)
    fused = copy.deepcopy(reference)
    input_ids = torch.randint(0, reference.config.vocab_size, (2, 19), device="cuda")
    lora_fuse.patch_lora_fuse()
    fused_forward = Linear.forward

    calls = []
    fused_function = lora_fuse.lora_fused

    def counted(*args, **kwargs):
        calls.append(None)
        return fused_function(*args, **kwargs)

    lora_fuse.lora_fused = counted
    try:
        Linear.forward = original
        reference_loss, reference_grads = _run_model(reference, input_ids)
        Linear.forward = fused_forward
        fused_loss, fused_grads = _run_model(fused, input_ids)
    finally:
        lora_fuse.lora_fused = fused_function
        Linear.forward = original

    assert len(calls) == 2  # q_proj and v_proj
    torch.testing.assert_close(fused_loss, reference_loss, rtol=2e-3, atol=2e-3)
    assert fused_grads.keys() == reference_grads.keys()
    for name in fused_grads:
        torch.testing.assert_close(fused_grads[name], reference_grads[name],
                                   rtol=2e-2, atol=2e-3)


def test_dropout_mask_is_reused_by_backward():
    from solution.lora_fuse import lora_fused

    torch.manual_seed(12)
    p = 0.05
    x = torch.ones(1024, 16, device="cuda", dtype=torch.float16, requires_grad=True)
    base = torch.zeros_like(x)
    a = torch.eye(16, device="cuda", dtype=torch.float32)
    b = torch.eye(16, device="cuda", dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.float16):
        out = lora_fused(x, base, a, b, 1.0, p)
    out.sum().backward()

    forward_keep = out != 0
    backward_keep = x.grad != 0
    assert torch.equal(forward_keep, backward_keep)
    assert abs(forward_keep.float().mean().item() - (1.0 - p)) < 0.01
    expected = torch.full_like(x.grad, 1.0 / (1.0 - p))
    torch.testing.assert_close(x.grad[backward_keep], expected[backward_keep],
                               rtol=1e-3, atol=1e-3)


def test_patched_peft_layer_retains_less_activation_memory(monkeypatch):
    from solution import lora_fuse

    monkeypatch.setenv("SOUP_LORA_FUSE", "1")
    LoraConfig, _, Linear = _peft()
    original = getattr(Linear.forward, "_soup_lora_fuse_original", Linear.forward)
    Linear.forward = original
    n, width, rank = 2048, 1536, 16
    x = torch.randn(n, width, device="cuda", dtype=torch.float16, requires_grad=True)
    base = torch.nn.Linear(width, width, bias=False).half().cuda()
    config = LoraConfig(r=rank, lora_alpha=32, lora_dropout=0.05,
                        target_modules=["proj"], bias="none")
    layer = Linear(base, "default", config, r=rank, lora_alpha=32).cuda().train()

    def retained_bytes(run):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        output = run()
        torch.cuda.synchronize()
        retained = torch.cuda.memory_allocated() - before
        del output
        gc.collect()
        torch.cuda.synchronize()
        return retained

    def peft_path():
        with torch.autocast("cuda", dtype=torch.float16):
            return layer(x)

    try:
        reference_bytes = retained_bytes(peft_path)
        lora_fuse.patch_lora_fuse()
        fused_forward = Linear.forward
        Linear.forward = fused_forward
        fused_bytes = retained_bytes(peft_path)
    finally:
        Linear.forward = original

    assert fused_bytes < reference_bytes
