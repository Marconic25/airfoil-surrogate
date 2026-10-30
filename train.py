import torch
import torch.nn as nn
from model import MLP
from data import AirfDataset, load_cached, split_by_simulation, normalization

# ordine della parte principale (da scrivere):
# 1) carica i dati e split per simulazione
# 2) stats = normalization(train_sims, ...)  → PRIMA dei Dataset, perché AirfDataset le riceve
# 3) train_ds (fixed=False) e val_ds (fixed=True), poi i DataLoader
# 4) model = MLP(...), criterion = nn.MSELoss(), optimizer = Adam(lr dal config)
# 5) ciclo sulle epoche: run_epoch(train, con optimizer) e run_epoch(val, senza)

def run_epoch(model, loader, criterion, optimizer=None):
    # un'epoca = un passaggio completo su tutti i dati del loader
    # criterion arriva da fuori (nn.MSELoss(), creata nella parte principale), come model e loader
    # optimizer=None → validation (pesi fermi); optimizer passato → training (pesi aggiornati)
    training = optimizer is not None
    model.train() if training else model.eval()     # eval() non cambia nulla per questa MLP, ma è buona abitudine
    total, n = 0.0, 0                               # somma delle loss (pesate) e numero di punti visti
    with torch.set_grad_enabled(training):          # in validation niente gradienti: meno memoria e più veloce
        for x,y in loader:                          # x: (B, K, 7), y: (B, K, 4), B simulazioni da K punti
            x = x.reshape(-1, x.shape[-1])          # (B*K, 7): la MLP guarda un punto alla volta
            y = y.reshape(-1, y.shape[-1])          # (B*K, 4)
            pred = model(x)                         # model è l'istanza passata, MLP(...) ne creerebbe una nuova
            loss = criterion(pred, y)               # MSE su target normalizzati, già media sul batch
            if training:                            # solo in training si aggiornano i pesi
                optimizer.zero_grad()               # azzera i gradienti del batch precedente
                loss.backward()                     # calcola i gradienti di questa loss
                optimizer.step()                    # aggiorna i pesi
            # fuori dall'if: serve anche in validation per ottenere la val loss (con n=0 si divide per zero)
            total += loss.item() * x.shape[0]       # loss del batch è una media: la pesiamo per il numero di punti
            n += x.shape[0]                         # così l'ultimo batch, se più piccolo, pesa giusto
    return total/n                                  # loss media sull'epoca (non media delle medie)
