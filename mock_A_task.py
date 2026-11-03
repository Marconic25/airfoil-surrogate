"""
MOCK - PARTE A (35 min): scrivi il loop da zero

Crea un file nuovo, mock_A_train.py, che fa `from mock_A_task import make_task`.
Niente AI, niente appunti, niente documentazione. Parli ad alta voce mentre scrivi.

TASK
  make_task() restituisce x [n,4] e y [n,2], float64.
  y ha due variabili con scale molto diverse (come pressione e velocita').
  Obiettivo: un MLP che predice y da x.

COSA DEVE FARE IL TUO SCRIPT
  1. Seed fissato (numpy e torch) e scelta del device.
  2. Split 70/15/15 train/val/test per indice, con permutazione.
  3. Statistiche di normalizzazione (media/std per colonna) calcolate SOLO sul
     train, per x e per y. Salvate su file (np.savez). Epsilon sul divisore.
  4. Dataset custom (__len__, __getitem__) che restituisce tensori float32.
  5. nn.Module MLP con config in un dict (d_in, d_hid, n_layers, d_out).
  6. Prima di tutto: funzione che prova l'overfit su un solo batch
     e stampa la loss finale. (Se non va a ~0, non si procede.)
  7. Loop di training: AdamW, MSE sui target normalizzati, clip_grad_norm_,
     scheduler cosine chiamato una volta per epoca, 40 epoche.
  8. Valutazione a ogni epoca con eval() e no_grad(); stampa train e val loss.
  9. Checkpoint del miglior val (modello + optimizer + epoca), poi ricaricato
     a fine training.
 10. Sul test, con il checkpoint ricaricato: RMSE per variabile in unita'
     ORIGINALI (denormalizza prima di calcolare l'errore).

NON FARE
  - non cercare niente online, non chiedere a nessuno;
  - non usare sklearn per split o normalizzazione.

Se a 35 minuti non hai finito: ti fermi e passi alla parte B. Annota cosa manca.
"""
import numpy as np


def make_task(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 4))
    y1 = np.sin(3 * x[:, 0]) + x[:, 1] ** 2 - 0.5 * x[:, 2]
    y2 = 200.0 * (x[:, 3] * x[:, 0] + 0.1 * rng.normal(size=n)) + 1000.0
    y = np.stack([y1, y2], axis=1)
    return x, y