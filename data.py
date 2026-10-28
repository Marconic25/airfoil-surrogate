import airfrans as af
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
#load the dataset
dataset, dataname = af.dataset.load(root = "data/Dataset", task = "scarce", train = True)

import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

input_cols = cfg["columns"]["input"]
output_cols = cfg["columns"]["output"]
val_fraction = cfg["split"]["val_fraction"]
seed = cfg["split"]["seed"]


class AirfDataset (Dataset):
    def __init__(self, sims, k, input_cols, output_cols): #qui passi tutto ciò che la classe deve ricordare
        self.sims = sims
        self.k = k
        self.input_cols = input_cols
        self.output_cols = output_cols

    def __len__(self): #restituisce il numero di simulazioni
        return len(self.sims)

    def __getitem__(self,i): #deve restituire K punti casuali della simulazione i, separando input e target
        sim = self.sims[i]                                              # array (N, 12)
        idx = np.random.choice(sim.shape[0], self.k, replace = False)   # K righe distinte tra 0 e N-1
        points = sim[idx]                                               # array (K, 12)
        x = points[:, self.input_cols]                                  # array (K, n_input)
        y = points[:, self.output_cols]                                 # array (K, n_output)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

def split_by_simulation(sims, names, val_fraction, seed):
    n = len(sims)
    n_val =  round(val_fraction*n)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = perm[0:n_val]
    train_idx = perm[n_val:]
    train_sims = []
    train_names = []
    val_sims = []
    val_names = []
    for j in train_idx:
        train_sims.append(sims[j])
        train_names.append(names[j])
    for i in val_idx:
        val_sims.append(sims[i])
        val_names.append(names[i])
    return train_sims, train_names, val_sims, val_names

def normalization(train_sims, input_cols, output_cols):
    col_train = np.concatenate(train_sims)
    x = col_train[:, input_cols]
    y = col_train[:, output_cols]
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    mean_y = y.mean(axis=0)
    std_y = y.std(axis=0)
    return mean_x, std_x, mean_y, std_y

def normalization(train_sims, input_cols, output_cols):
    col_train = np.concatenate(train_sims)
    x = col_train[:, input_cols]
    y = col_train[:, output_cols]
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    mean_y = y.mean(axis=0)
    std_y = y.std(axis=0)
    stats = {"mean_x": mean_x, "std_x": std_x, "mean_y": mean_y, "std_y": std_y}
    return stats

 1) split train/val per simulazione (dataset e dataname vengono dalla cella di caricamento)
train_sims, train_names, val_sims, val_names = split_by_simulation(dataset, dataname, val_fraction, seed)

stats = normalization(small_sims, input_cols, output_cols)

def save_stats(stats, path):
    folder = os.path.dirname(path)              # cartella del file, es. "artifacts"
    if folder:                                  # se path è solo un nome, folder è "" e non serve crearla
        os.makedirs(folder, exist_ok=True)
    np.savez(path, **stats)                     # ogni chiave del dizionario diventa un array con nome nel file

def load_stats(path):
    f = np.load(path)                           # oggetto che si legge come un dizionario, ma non lo è
    stats = {}
    for key in f.files:                         # f.files = elenco dei nomi salvati
        stats[key] = f[key]
    return stats