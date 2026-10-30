import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

D_i, D_k, D_o = 10, 40, 5

model = nn.Sequential(
    nn.Linear(D_i, D_k),
    nn.ReLU()
    nn.Linear(D_k, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_o)
)

def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight)
        layer_in.bias.data.fill(0.0)
model.aplly(weights_init)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
scheduler = steLR(optimizer, step_size=10, gamma = 0.5)

x = torch.randn(100, D_i)
y = torch.randn(100, D_o)

data_loader = DataLoader(TensorDataset(x, y), batch_size = 10, shuffle = True)

for epoch in range(100):
    epoch_loss = 0.0
    for i, data in enumerate(data_loader)
    x_batch, y_batch = data
    optimizer.zero_grad()
    pred = model(x_batch)
    loss = criterion(pred, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    print(epoch, loss)
    scheduler.step