import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

torch.manual_seed(0)
rng = np.random.default_rng(0)

n = 4000
x = rng.uniform(-2, 2, size=(n,3)).astype(np.float32)
y = (np.sin(x[:, 0]) + x[:, 1] ** 2 + 0.5 * x[:, 2]).astype(np.float32)
y = y[:, None] #crea colonna nuova

idx = rng.permutation(n)
n_train = int(0.8*n)
tr, va = idx[:n_train], idx[n_train:]

#stats
mean = x[tr].mean(axis=0)
std = x[tr].std(axis=0)
x = (x-mean)/std

class PointDataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class Net(nn.Module):
    def __init__(self, d_in = 3, d_hid = 64, d_out=1, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_out)
        )

    def forward(self, x):
        return self.net(x)

train_ds = PointDataset(x[tr], y[tr])
va_ds = PointDataset(x[va], y[va])
train_loader = DataLoader(train_ds, batch_size = 128, shuffle=True)
val_loader = DataLoader(va_ds, batch_size = 128, shuffle=False)

model= Net()
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr = 1e-2)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma = 0.5)

train_losses, val_losses = [], []

for epoch in range(30):
    model.train()
    tot = 0.0
    for xb, yb in train_loader:
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(yb, pred)
        loss.backward()
        opt.step()
        tot+= loss.item()*len(xb)
    sched.step()
    train_loss = tot/len(train_ds)

    tot=0.0
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = loss_fn(yb, pred)
            tot+= loss.item()*len(xb)
    val_loss = tot/len(va_ds)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"epoch{epoch:02d} train{train_loss:4f} val{val_loss:4f}")


fig, ax = plt.subplots()
ax.plot(train_losses, label="train")
ax.plot(val_losses, label="val")
ax.set_xlabel("epoca")
ax.set_ylabel("MSE")
ax.set_yscale("log")
ax.legend()
fig.savefig("loss.png")
plt.show()




