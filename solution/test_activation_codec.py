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
