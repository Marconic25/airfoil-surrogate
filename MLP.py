#example of NN from Understanding Deep Learning
#creates random input and output datasets and inizializes the wrights
#random input and outputs
#least square loss
#SGD with momentum in batches of size 10 over 100 epochs
#lr starts at 0.01 and halves every 10 epochs

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn


#define input size, hidden layer size, putput size
D_i, D_k, D_o = 10, 40, 5

#create a model with two hidden layers
model = nn.Sequential(
    nn.Linear(D_i, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_o)
)

#He inizialization of the weights
def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear): #ReLu e sequential non hanno pesi, la funzione agisce solo sui layer Linear
        nn.init.kaiming_normal_(layer_in.weight) #He
        layer_in.bias.data.fill_(0.0) #bias a zero
model.apply(weights_init) #apply scorre ricorsivamente tutti i sottomoduli del modello e applica
#weights_init su ognuno

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
scheduler = StepLR(optimizer, step_size=10, gamma = 0.5) #gamma = 0.5 vuol dire halves every 10

x = torch.randn(100, D_i)
y = torch.randn(100, D_o)

data_loader = DataLoader(TensorDataset(x,y), batch_size = 10, shuffle=True)

for epoch in range(100):
    epoch_loss = 0.0
    for i, data in enumerate(data_loader):
        x_batch, y_batch = data
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss. backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch:5d}, loss {epoch_loss:.3f}')
# tell scheduler to consider updating learning rate
    scheduler.step() #update del lr ogni epoch quindi sta nel ciclo delle epoch