import numpy as np
import torch
import pytest

from data import AirfDataset, normalization, split_by_simulation, normalize_x

import yaml
with open("./config.yaml") as f:
    cfg = yaml.safe_load(f)
val_fraction = cfg["split"]["val_fraction"]
seed = cfg["split"]["seed"]
N_SIMS = 10
N_POINTS = 200
N_COLS = 12
K = 50
INPUT_COLS = [0, 1, 2, 3, 4, 5, 6]
OUTPUT_COLS = [7, 8, 9]

@pytest.fixture #funzione decorata che restituisce sim finte
def sims():
    rng = np.random.default_rng(0) #crea generatore numeri casuali
    #ritorniamo un array di numeri casuali con distribuzione normale
    return [rng.normal(size=(N_POINTS,N_COLS)) for _ in range(N_SIMS)]
    #ciclo ripete questa cosa N volte ritornando una lista di N_SIMS di array numpy.
    #ognuno di shape (N_POINTS,N_COL)

@pytest.fixture
def name_sims():
    return [f"sim_{i}" for i in range(N_SIMS)]

#questa restituisce un AirfDataset pronto all'uso, partendo dalle sim finte
@pytest.fixture
def ds(sims):
    stats = normalization(sims, INPUT_COLS, OUTPUT_COLS) #calcola le stats, che servono a __getitem__ per normalizzare
    return AirfDataset(sims, K, INPUT_COLS, OUTPUT_COLS, stats)



def test_shape(ds):
    x, y = ds[0] #chiama __getitem__(0) che prende 50 punti a caso della prima sim
    #restituisce così (x, y)

    #tipo1: entrambi tensori torch
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    #tipo2: shape(K, n_input) e (K. n_output)
    assert x.shape == (K,len(INPUT_COLS))
    assert y.shape == (K, len(OUTPUT_COLS))

    #tipo3: float32
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32

    #tipo4: lunghezza dataset
    assert len(ds) == N_SIMS


def test_mean(sims, name_sims):
    train_sims, train_names, val_sims, val_names = split_by_simulation(sims, name_sims, val_fraction, seed)
    stats = normalization(train_sims, INPUT_COLS, OUTPUT_COLS)

    x_train = np.concatenate(train_sims)[:, INPUT_COLS]
    x_norm = normalize_x(x_train, stats)

    assert np.allclose(x_norm.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(x_norm.std(axis=0), 1)

