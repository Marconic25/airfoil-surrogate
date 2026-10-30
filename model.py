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

    @staticmethod   # metodo senza self: non usa l'oggetto, riceve solo il layer. apply() lo chiama su ogni sottomodulo
    def _init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)
            layer.bias.data.fill_(0.0)


    def forward(self,x):
        # il DataLoader dà x di forma (B, K, 7): B simulazioni, K punti ciascuna, 7 feature per punto
        # la MLP lavora un punto alla volta, quindi prima di chiamarla si appiattisce a (B*K, 7):
        #   x = x.reshape(-1, x.shape[-1])    # -1 = calcola tu le righe (B*K)
        # stesso discorso per y: (B, K, 4) -> (B*K, 4). Cambia solo l'organizzazione, non i numeri
        return self.net(x)


