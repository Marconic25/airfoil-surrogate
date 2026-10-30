import airfrans as af
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
#load the dataset
#dataset, dataname = af.dataset.load(root = "data/Dataset", task = "scarce", train = True)


import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

input_cols = cfg["columns"]["input"]
output_cols = cfg["columns"]["output"]
val_fraction = cfg["split"]["val_fraction"]
seed = cfg["split"]["seed"]


class AirfDataset (Dataset):
    def __init__(self, sims, k, input_cols, output_cols, stats, fixed= False, seed=0): #qui passi tutto ciò che la classe deve ricordare
        self.sims = sims
        self.k = k
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.stats = stats
        self.fixed = fixed
        self.rng = np.random.default_rng(seed)
        if fixed:
            self.fixed_idx = [self.rng.choice(sim.shape[0], k, replace = False) for sim in sims]

    def __len__(self): #restituisce il numero di simulazioni
        return len(self.sims)

    def __getitem__(self,i): #deve restituire K punti casuali della simulazione i, separando input e target
        sim = self.sims[i]     
        if self.fixed:       
            idx = self.fixed_idx[i]
        else:                                  # array (N, 12)
            idx = self.rng.choice(sim.shape[0], self.k, replace = False)   # K righe distinte tra 0 e N-1
        points = sim[idx]                                               # array (K, 12)
        x = points[:, self.input_cols]                                  # array (K, n_input)
        y = points[:, self.output_cols]                                 # array (K, n_output)
        x = normalize_x(x, self.stats)   
        y = normalize_y(y, self.stats)                               #normalizzazione direttamente qua dentro
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
    stats = {"mean_x": mean_x, "std_x": std_x, "mean_y": mean_y, "std_y": std_y}
    return stats

def normalize_x(x, stats):
    x_norm = (x - stats["mean_x"]) / stats["std_x"]
    return x_norm

def normalize_y(y, stats):
    y_norm = (y - stats["mean_y"]) / stats["std_y"]
    return y_norm

def denormalize_y(y_norm, stats):
    y_denorm = stats["std_y"]* y_norm + stats["mean_y"]
    return y_denorm

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


    
def load_cached(path, n_sims=None):
    if os.path.exists(path):                    # già salvato: lo rileggo, pochi secondi
        with open(path, "rb") as f:
            return pickle.load(f)
    dataset, dataname = af.dataset.load(root="data/Dataset", task="scarce", train=True)   # lento, solo la prima volta
    if n_sims is not None:                      # per sviluppare tengo solo le prime n_sims simulazioni
        dataset = dataset[:n_sims]
        dataname = dataname[:n_sims]
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "wb") as f:                 # salvo la coppia (dataset, dataname)
        pickle.dump((dataset, dataname), f)
    return dataset, dataname

if __name__ == "__main__":
    # 1) caricamento: la prima volta lento (2-3 minuti), poi dalla cache
    dataset, dataname = load_cached("data/cache_dev.pkl", n_sims=30)

    # 2) split train/val
    train_sims, train_names, val_sims, val_names = split_by_simulation(dataset, dataname, val_fraction, seed)
    print("train:", len(train_sims), "val:", len(val_sims))

    # 3) sottoinsieme piccolo e statistiche
    small_sims = train_sims[0:10]
    stats = normalization(small_sims, input_cols, output_cols)
    save_stats(stats, "artifacts/stats.npz")

    print(stats.keys())
    print(stats["mean_x"].shape, stats["mean_y"].shape)

    # 4) un campione dal Dataset
    ds = AirfDataset(small_sims, 1000, input_cols, output_cols, stats)
    x, y = ds[0]
    print(x.shape, y.shape)
    print(x.mean(dim=0), x.std(dim=0))

