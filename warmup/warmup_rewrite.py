import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#torch              → tensori, autograd (il tuo vecchio script usava solo questo)
#torch.nn           → modello e loss (nn.Module, nn.Linear, nn.MSELoss)
#torch.optim        → ottimizzatori (SGD)
#torch.utils.data   → Dataset e DataLoader

class CLDataset(Dataset): #senza () crea classe vuota, mentre con dentro dataset parte da quello che sa già fare
    def __init__(self):
        alpha_deg = torch.linspace(-5,20,25)
        self.x = torch.deg2rad(alpha_deg)
        cl_lin = torch.pi*2*self.x
        cl_clamped = torch.clamp(cl_lin, max = 1.6449)
        self.y = 0.05*torch.randn_like(cl_clamped) + cl_clamped
        self.x = self.x.unsqueeze(1)
        self.y = self.y.unsqueeze(1)

    def __len__(self): #serve perchè DataLoader ce l ha bisogno
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class LinearModel(nn.Module): #nn.Module contiene tutto
    def __init__(self):
        super().__init__() #esegui quello del genitore e anche il mio
        self.linear = nn.Linear(1, 1) #contiene solo il singolo layer (input, outputs) quindi quanti pesi ha il layer

    def forward(self, x): #init crea i layer ma gli devi dire come si collegano
        return self.linear(x)

#inizializzi modello,loss,optimizer
model = LinearModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

ds = CLDataset()
loader = DataLoader(ds, batch_size=5, shuffle=True) #raggruppa in patch e fa cose che il dataset non fa
N = 1000
loss_hist = []
for epoch in range(N):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad() #rimette a zero il gradiente
        loss.backward() #calcola la loss backward
        optimizer.step() #fa un passo di correzione
        loss_hist.append(loss.item())

print(model.linear.weight.item(), model.linear.bias.item())





