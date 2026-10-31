"""Forward stays exact; compressed backward storage has bounded derivative error."""
import pytest
import torch
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='needs CUDA')

@pytest.mark.parametrize('shape',[(9,127),(2,31,1536)])
def test_saved_norm_fp16(monkeypatch,shape):
 from solution.add_rmsnorm import add_rmsnorm
 torch.manual_seed(32)
 res=torch.randn(shape,device='cuda',dtype=torch.float32,requires_grad=True)
 delta=torch.randn(shape,device='cuda',dtype=torch.float16,requires_grad=True)
 w=torch.randn(shape[-1],device='cuda',dtype=torch.float32)
 dy=torch.randn(shape,device='cuda',dtype=torch.float16)
 ds=torch.randn(shape,device='cuda',dtype=torch.float32)
 cases=[]
 for flag in ['0','1']:
  monkeypatch.setenv('SOUP_X_NORM_SAVE16',flag)
  saved=[]
  def pack(t):
   saved.append(t.numel()*t.element_size())
   return t
  with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
   y,total=add_rmsnorm(res,delta,w,1e-6,torch.float16)
  grads=torch.autograd.grad((y,total),(res,delta),(dy,ds))
  cases.append((y,total,grads,sum(saved)))
 for a,b in zip(cases[0][:2],cases[1][:2]):torch.testing.assert_close(a,b,atol=0,rtol=0)
 for a,b in zip(cases[0][2],cases[1][2]):torch.testing.assert_close(a,b,atol=2e-3,rtol=2e-3)
 assert cases[1][3]<cases[0][3]*.7
