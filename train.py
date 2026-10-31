import torch
import torch.nn as nn
from model import MLP
from data import AirfDataset, load_cached, split_by_simulation, normalization, save_stats
import yaml
import numpy as np
from torch.utils.data import DataLoader
import os 
import csv

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



if __name__ == "__main__":
    # config
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    input_cols  = cfg["columns"]["input"]
    output_cols = cfg["columns"]["output"]
    k = cfg["dataset"]["k_points"]

    # seed fisso: rende riproducibili pesi iniziali e campionamento
    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    val_fraction = cfg["split"]["val_fraction"]
    split_seed = cfg["split"]["seed"]


    # 1) carica e split per simulazione
    dataset, dataname = load_cached("data/cache_dev.pkl", n_sims=cfg["dataset"]["n_sims"])
    train_sims, train_names, val_sims, val_names = split_by_simulation(dataset, dataname, val_fraction, split_seed)

    # 2) statistiche calcolate SOLO sul train
    stats = normalization(train_sims, input_cols, output_cols)

    # 3) dataset e loader
    train_ds = AirfDataset(train_sims, k, input_cols, output_cols, stats, fixed=False, seed=seed)
    val_ds   = AirfDataset(val_sims, k, input_cols, output_cols, stats, fixed=True, seed=seed)

    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False) #con val non serve mescolare


    # 4) modello, loss, ottimizzatore
    in_dim = len(input_cols)
    out_dim = len(output_cols)                                           
    model = MLP(in_dim, out_dim, cfg["model"]["hidden"], cfg["model"]["depth"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["train"]["step_size"], gamma=cfg["train"]["gamma"])
    # training: ciclo sulle epoche, con log CSV
    os.makedirs("runs", exist_ok=True)
    epochs = cfg["train"]["epochs"]

    best_val = float("inf")
    save_stats(stats, "runs/stats.npz")
    with open("runs/log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch in range(epochs):
            train_loss = run_epoch(model, train_loader, criterion, optimizer)
            val_loss = run_epoch(model, val_loader, criterion, optimizer = None)
            writer.writerow([epoch, train_loss, val_loss])
            f.flush()                           # scrive subito su disco, utile se interrompi
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "cfg": cfg,
            "epoch": epoch, "val_loss": val_loss}, "runs/best.pt")
        scheduler.step()

                
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")


                                      

