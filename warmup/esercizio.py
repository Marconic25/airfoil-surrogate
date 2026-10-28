import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class CLDataset(Dataset):
    def __init__(self):
        alpha_deg = torch.linspace(-5,20,25)
        self.x = torch.deg2rad(alpha_deg)
        cl_lin = 2*torch.pi*self.x
        self.y = torch.clamp(0.05*torch.randn_like(cl_lin) + cl_lin, max = 1.6449)
        self.x = self.x.unsqueeze(1)
        self.y = self.y.unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self,i):
        return self.x[i], self.y[i]

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self,x):
        return self.linear(x)

N = 1000
ds = CLDataset()
loader = DataLoader(ds, batch_size = 5)

model = LinearModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
loss_hist = []
for epoch in range(N):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())

print(model.linear.weight.item())



    
    

    
        

   






