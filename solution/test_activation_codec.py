"""GPU tests: exact forward, bounded gradient error, actual saved-memory reduction."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

@pytest.mark.parametrize('shape', [(7, 113), (23, 1091), (2, 31, 8960)])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_codec_values_gradients_and_saved_bytes(shape, dtype):
    from solution.activation_codec import SavedActivationInt8
    from solution.swiglu import swiglu
    torch.manual_seed(13)
    g = torch.randn(shape, device='cuda', dtype=dtype, requires_grad=True)
    u = torch.randn(shape, device='cuda', dtype=dtype, requires_grad=True)
    grad = torch.randn_like(g)
    results = []
    for function in [swiglu, SavedActivationInt8.apply]:
        saved = []
        def pack(tensor):
            saved.append(tensor.numel() * tensor.element_size())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            out = function(g, u)
        gradients = torch.autograd.grad(out, (g, u), grad)
        results.append((out, gradients, sum(saved)))
    # Transcendental operation spelling may cause a last-bit forward difference.
    torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-5 if dtype==torch.float32 else 1e-3, atol=1e-6 if dtype==torch.float32 else 1e-3)
    for ref, compressed in zip(results[0][1], results[1][1]):
        relative_rms = (ref.float()-compressed.float()).square().mean().sqrt() / ref.float().square().mean().sqrt()
        assert relative_rms.item() < .02
    assert results[1][2] < results[0][2] * .55


@pytest.mark.parametrize('shape', [(7, 113), (2, 31, 8960)])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('distribution', ['normal', 'small_gate', 'block_scales'])
def test_int4_exact_forward_bounded_backward_and_saved_bytes(shape, dtype, distribution):
    from solution.activation_codec import SavedActivationInt4, SavedActivationInt8
    from solution.swiglu import swiglu

    torch.manual_seed(73)
    g = torch.randn(shape, device='cuda', dtype=dtype)
    u = torch.randn_like(g)
    if distribution == 'small_gate':
        # SiLU'(0)=1/2: quantization must preserve its gradient near zero.
        g.mul_(0.03)
    elif distribution == 'block_scales':
        blocks = (g.numel() + 127) // 128
        scales = torch.tensor([0., 0.001, 0.1, 1., 12.], device='cuda')
        scales = scales[torch.arange(blocks, device='cuda') % scales.numel()]
        scales = scales.repeat_interleave(128)[:g.numel()].reshape(shape).to(dtype)
        g.mul_(scales)
        u.mul_(scales)
    g.requires_grad_(True)
    u.requires_grad_(True)
    grad = torch.randn_like(g)
    results = []
    for function in [swiglu, SavedActivationInt8.apply, SavedActivationInt4.apply]:
        saved = []

        def pack(tensor):
            saved.append(tensor.numel() * tensor.element_size())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            out = function(g, u)
        gradients = torch.autograd.grad(out, (g, u), grad)
        results.append((out, gradients, sum(saved)))

    atol = 1e-6 if dtype == torch.float32 else 1e-3
    rtol = 1e-5 if dtype == torch.float32 else 1e-3
    torch.testing.assert_close(results[2][0], results[0][0], rtol=rtol, atol=atol)
    for expected, actual in zip(results[0][1], results[2][1]):
        assert torch.isfinite(actual).all()
        relative_rms = ((expected.float() - actual.float()).square().mean().sqrt()
                        / expected.float().square().mean().sqrt().clamp_min(1e-8))
        # This is an intentionally lossy opt-in storage mode. End-to-end Soup
        # loss/quality qualification is required separately before enabling it.
        assert relative_rms.item() < .18
    assert results[2][2] < results[0][2] * .30
    assert results[2][2] < results[1][2] * .56


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_int4_odd_tail_zero_blocks_and_signs(dtype):
    from solution.activation_codec import SavedActivationInt4
    from solution.swiglu import swiglu

    values = torch.tensor([0., 0.07, -0.15, 0.25, -0.37, 0.52, -0.72, 1., -1.],
                          device='cuda', dtype=dtype)
    # Entire first block is zero; the final odd byte has one unused nibble.
    g = torch.cat([torch.zeros(128, device='cuda', dtype=dtype), values]).requires_grad_()
    u = torch.cat([torch.zeros(128, device='cuda', dtype=dtype), values.flip(0)]).requires_grad_()
    expected = swiglu(g, u)
    actual = SavedActivationInt4.apply(g, u)
    grad = torch.ones_like(g)
    eg = torch.autograd.grad(expected, (g, u), grad)
    ag = torch.autograd.grad(actual, (g, u), grad)
    tolerance = 2e-3 if dtype == torch.float16 else 1e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for ref, compressed in zip(eg, ag):
        torch.testing.assert_close(compressed, ref, rtol=tolerance, atol=tolerance)
        assert torch.count_nonzero(compressed[:128]) == 0


def test_int4_patch_selects_opt_in_and_keeps_trainable_mlp_fallback(monkeypatch):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from solution import activation_codec

    current = Qwen2MLP.forward
    original = getattr(current, '_soup_activation_original', current)
    monkeypatch.setattr(Qwen2MLP, 'forward', original)
    monkeypatch.delenv('SOUP_X_ACT_INT8', raising=False)
    monkeypatch.delenv('SOUP_X_ACT_INT4', raising=False)
    activation_codec.patch_saved_activation_int8()
    assert Qwen2MLP.forward is original

    monkeypatch.setenv('SOUP_X_ACT_INT4', '1')
    activation_codec.patch_saved_activation_int8()
    assert Qwen2MLP.forward._soup_saved_codec == 'int4'
    layer = Qwen2MLP(Qwen2Config(hidden_size=32, intermediate_size=64)).cuda().half().train()
    x = torch.randn(3, 32, device='cuda', dtype=torch.float16, requires_grad=True)
    calls = []
    real_apply = activation_codec.SavedActivationInt4.apply

    def observed(*args):
        calls.append(True)
        return real_apply(*args)

    monkeypatch.setattr(activation_codec.SavedActivationInt4, 'apply', observed)
    expected = original(layer, x)
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    assert not calls  # Trainable MLP weights must keep the original backward.
    layer.requires_grad_(False)
    layer(x).sum().backward()
    assert calls and x.grad is not None and torch.isfinite(x.grad).all()
