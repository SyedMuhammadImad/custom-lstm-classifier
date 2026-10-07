import torch
import pytest
import models as m

def test_custom_cell_matches_reference_values_and_input_state_gradients():
    torch.manual_seed(4);custom=m.MyLSTMCell(3,4).double();ref=torch.nn.LSTMCell(3,4).double()
    # Local f,i,g,o ordering -> PyTorch i,f,g,o; local combined input is h,x.
    order=torch.cat([torch.arange(4,8),torch.arange(0,4),torch.arange(8,16)])
    with torch.no_grad():
        ref.weight_ih.copy_(custom.gates.weight[order,4:]);ref.weight_hh.copy_(custom.gates.weight[order,:4])
        ref.bias_ih.copy_(custom.gates.bias[order]);ref.bias_hh.zero_()
    originals=[torch.randn(2,n,dtype=torch.float64,requires_grad=True) for n in (3,4,4)]
    clones=[a.detach().clone().requires_grad_() for a in originals]
    a=custom(*originals);b=ref(clones[0],(clones[1],clones[2]))
    for u,v in zip(a,b):torch.testing.assert_close(u,v)
    sum(v.square().sum() for v in a).backward();sum(v.square().sum() for v in b).backward()
    for u,v in zip(originals,clones):torch.testing.assert_close(u.grad,v.grad)
    torch.testing.assert_close(custom.gates.weight.grad[order,4:],ref.weight_ih.grad)
    torch.testing.assert_close(custom.gates.weight.grad[order,:4],ref.weight_hh.grad)
    torch.testing.assert_close(custom.gates.bias.grad[order],ref.bias_ih.grad)
    assert torch.autograd.gradcheck(custom,tuple(originals),fast_mode=True)

def test_stacked_sequence_is_equivalent_to_explicit_cell_unroll():
    torch.manual_seed(8);net=m.MyLSTM(2,3,2).double();x=torch.randn(2,7,2,dtype=torch.float64,requires_grad=True)
    h=[torch.zeros(2,3,dtype=torch.float64) for _ in range(2)];c=[v.clone() for v in h]
    for t in range(7):
        v=x[:,t]
        for layer,cell in enumerate(net.cells):h[layer],c[layer]=cell(v,h[layer],c[layer]);v=h[layer]
    torch.testing.assert_close(net(x),h[-1]);net(x).sum().backward();assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(net(x.detach()),net(x.detach()))
    with pytest.raises(ValueError):net(torch.empty(2,0,2,dtype=torch.float64))
    with pytest.raises(ValueError):m.MyLSTM(2,3,0)
