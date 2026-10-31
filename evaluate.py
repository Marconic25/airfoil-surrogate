import numpy as np
import torch
import yaml
import csv
from model import MLP
from data import load_cached, split_by_simulation, load_stats, normalize_x, denormalize_y
import matplotlib.pyplot as plt
import airfrans as af

def predict(model, sim, stats, input_cols):
    x = normalize_x(sim[:,input_cols], stats) #normalized inputs
    with torch.no_grad():
        p = model(torch.from_numpy(x).float()).numpy()
    return denormalize_y(p, stats)

def errors_by_region(model, sims, stats, input_cols, output_cols):
    sq = {"surface": 0.0, "volume": 0.0}                     # somma degli errori quadratici, per variabile
    n = {"surface": 0, "volume": 0}                          # numero di punti

    for sim in sims:
        pred = predict(model, sim, stats, input_cols)
        true = sim[:, output_cols]
        mask = sim[:, 11] > 0.5 #True sui punti di superficie
        err2 = (pred - true)**2
        sq["surface"] += err2[mask].sum(axis=0)
        n["surface"] += mask.sum()
        sq["volume"] += err2[~mask].sum(axis=0)
        n["volume"] += (~mask).sum()                         # ~mask = punti di volume
    return {k: sq[k] / n[k] for k in sq}                     # MSE per variabile, in ciascuna regione

import matplotlib.pyplot as plt

def cp_surface(sim, pred):                           # pred: (N, 4) in unità fisiche, da predict()
    surf = sim[:, 11] > 0.5
    u_inf = np.hypot(sim[0, 2], sim[0, 3])           # velocità all'infinito: costante in tutta la simulazione
    x = sim[surf, 0]
    cp_true = sim[surf, 9] / (0.5 * u_inf**2)        # colonna 9 = p vera
    cp_pred = pred[surf, 2]/(0.5*u_inf**2)
    return x, cp_true, cp_pred

def plot_cp(x, cp_true, cp_pred, title, path):
    plt.figure()
    plt.scatter(x, -cp_true, s=4, label="vero")
    plt.scatter(x, -cp_pred, s=4, label="previsto")
    plt.xlabel("x/c"); plt.ylabel("-Cp"); plt.legend(); plt.title(title)
    plt.savefig(path, dpi=150)                        # savefig: in WSL non c'è finestra

import airfrans as af

def coefficients(name, pred, root="data/Dataset"):
    S = af.Simulation(root=root, name=name)
    ref = S.force_coefficient(reference=True)          # ((cd, cdp, cdv), (cl, clp, clv)) dal file
    S.velocity = pred[:, 0:2]                          # u e v, (N, 2)
    S.pressure = pred[:, 2:3]                          # p, (N, 1): 2:3 e non 2, la libreria vuole una colonna
    S.nu_t     = pred[:, 3:4]                          # nut, (N, 1)
    est = S.force_coefficient()                        # stesso calcolo, ma con i campi previsti
    return ref, est



if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    input_cols = cfg["columns"]["input"]
    output_cols = cfg["columns"]["output"]

    # stesso split del training: stesse simulazioni di validation
    dataset, dataname = load_cached("data/cache_dev.pkl", n_sims=cfg["dataset"]["n_sims"])
    _, _, val_sims, val_names = split_by_simulation(dataset, dataname, cfg["split"]["val_fraction"], cfg["split"]["seed"])

    # statistiche e pesi salvati dal training
    stats = load_stats("runs/stats.npz")
    ckpt = torch.load("runs/best.pt")
    model = MLP(len(input_cols), len(output_cols), cfg["model"]["hidden"], cfg["model"]["depth"])
    model.load_state_dict(ckpt["model"]) 
    model.eval()                
    print("checkpoint dall'epoca", ckpt["epoch"], "con val", round(ckpt["val_loss"], 4))

    errs = errors_by_region(model, val_sims, stats, input_cols, output_cols)
    names = ["u", "v", "p", "nut"]
    for region, mse in errs.items():                     # RMSE in unità fisiche, stessa unità della variabile
        print(region, {n: round(float(v), 4) for n, v in zip(names, np.sqrt(mse))})
    sim = val_sims[0]
    pred = predict(model, sim, stats, input_cols)
    x, cp_t, cp_p = cp_surface(sim, pred)
    plot_cp(x, cp_t, cp_p, val_names[0], "runs/cp_0.png")

    # coefficienti Cd e Cl su tutte le simulazioni di validation: veri (dal file) contro previsti
    rows = []
    for name, sim in zip(val_names, val_sims):
        pred = predict(model, sim, stats, input_cols)
        ref, est = coefficients(name, pred)
        (cd_t, cdp_t, _), (cl_t, clp_t, _) = ref         # (totale, pressione, viscoso): il viscoso non si usa
        (cd_p, cdp_p, _), (cl_p, clp_p, _) = est
        rows.append([cd_t, cd_p, cdp_t, cdp_p, cl_t, cl_p, clp_t, clp_p])
        print("fatto", name, flush=True)
    rows = np.array(rows)                                # (20, 8)

    cols = ["cd_true", "cd_pred", "cdp_true", "cdp_pred", "cl_true", "cl_pred", "clp_true", "clp_pred"]
    with open("runs/coefficients.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"] + cols)
        for name, r in zip(val_names, rows):
            writer.writerow([name] + list(r))

    for k in ("cd", "cdp", "cl", "clp"):
        t = rows[:, cols.index(k + "_true")]
        p = rows[:, cols.index(k + "_pred")]
        print(k, "| errore assoluto medio", round(float(np.abs(p - t).mean()), 4),
              "| errore relativo medio", round(float((np.abs(p - t) / (np.abs(t) + 1e-12)).mean()), 3),
              "| correlazione vero-previsto", round(float(np.corrcoef(t, p)[0, 1]), 3))

