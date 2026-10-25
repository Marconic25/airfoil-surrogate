import torch

def test_torch_works():
    x = torch.ones(3)
    assert x.sum().item() == 3.0