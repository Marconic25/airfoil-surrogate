import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, depth):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)
            layer.bias.data.fill_(0.0)


    def forward(self,x):
        return self.net(x)


m = MLP(7,4,138,4)
print(m(torch.randn(10, 7)).shape)