"""
MOCK - PARTE B (25 min): trova i bug

Aprila solo quando hai finito (o fermato) la parte A.
Questo script contiene 3 bug. Puoi eseguirlo. Per ogni bug:
  - dimmi dove e' (riga), cosa causa, come lo correggi;
  - se puoi, dimmi come l'avresti notato dai numeri.
Niente AI, niente appunti. Parla ad alta voce.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
rng = np.random.default_rng(0)

# --- dati ---
n = 4000
x = rng.uniform(-2, 2, size=(n, 3)).astype(np.float32)
y = (np.sin(x[:, 0]) + x[:, 1] ** 2 + 0.5 * x[:, 2]).astype(np.float32)
y = y[:, None] #crea colonna nuova

idx = rng.permutation(n)
n_train = int(0.8 * n)
tr, va = idx[:n_train], idx[n_train:]

# statistiche calcolate solo sul train, applicate a tutto (niente leakage)
mean, std = x[tr].mean(axis=0), x[tr].std(axis=0)
x = (x - mean) / std


class PointDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class Net(nn.Module):
    def __init__(self, d_in=3, d_hid=64, d_out=1, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hid, d_out),
        )

    def forward(self, x):
        return self.net(x)


train_ds = PointDataset(x[tr], y[tr])
val_ds = PointDataset(x[va], y[va])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

model = Net()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
loss_fn = nn.MSELoss()

for epoch in range(30):
    model.train()
    tot = 0.0
    for xb, yb in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(xb)
    sched.step()
    train_loss = tot / len(train_ds)

    tot = 0.0
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            loss = loss_fn(model(xb), yb)
            tot += loss.item() * len(xb)
    val_loss = tot / len(val_ds)

    print(f"epoch {epoch:02d}  train {train_loss:.4f}  val {val_loss:.4f}")