import csv
import numpy as np
import matplotlib.pyplot as plt

# legge i CSV prodotti da evaluate.py e costruisce tabella (runs/results.md) e grafici (runs/*.png)
# il grafico 1 (Cp) lo scrive già evaluate.py in runs/cp_0.png

def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))

errs = read("runs/errors_by_region.csv")
coef = read("runs/coefficients.csv")
variables = ["u", "v", "p", "nut"]

def col(rows, key):
    return np.array([float(r[key]) for r in rows])

def err(region, variable, key):
    return [float(r[key]) for r in errs if r["region"] == region and r["variable"] == variable][0]

# grafico 2: Cl vero contro previsto, un punto per simulazione
cl_t, cl_p = col(coef, "cl_true"), col(coef, "cl_pred")
plt.figure()
plt.scatter(cl_t, cl_p, s=20)
lim = [min(cl_t.min(), cl_p.min()) - 0.1, max(cl_t.max(), cl_p.max()) + 0.1]
plt.plot(lim, lim, "k--", lw=1, label="y = x (previsione perfetta)")
plt.xlabel("Cl vero"); plt.ylabel("Cl previsto"); plt.legend()
plt.title("Cl sulle simulazioni di validation")
plt.savefig("runs/cl_scatter.png", dpi=150)
plt.close()

# grafico 3: errore per variabile in unità di deviazione standard, superficie contro volume
x = np.arange(len(variables)); w = 0.35
plt.figure()
plt.bar(x - w/2, [err("surface", v, "rmse_norm") for v in variables], w, label="superficie")
plt.bar(x + w/2, [err("volume", v, "rmse_norm") for v in variables], w, label="volume")
plt.axhline(1.0, color="k", ls="--", lw=1, label="predire sempre la media")
plt.xticks(x, variables); plt.ylabel("RMSE / deviazione standard"); plt.legend()
plt.title("Errore per variabile")
plt.savefig("runs/errors_by_variable.png", dpi=150)
plt.close()

# tabella
lines = ["# Risultati sulla validation", ""]
lines += ["## Errore per variabile", "",
          "| variabile | RMSE superficie | RMSE volume | superficie / std | volume / std |",
          "|---|---|---|---|---|"]
for v in variables:
    lines.append(f"| {v} | {err('surface', v, 'rmse'):.4g} | {err('volume', v, 'rmse'):.4g} "
                 f"| {err('surface', v, 'rmse_norm'):.2f} | {err('volume', v, 'rmse_norm'):.2f} |")

lines += ["", "## Coefficienti aerodinamici", "",
          "| coefficiente | errore assoluto medio | correlazione | pendenza previsto contro vero |",
          "|---|---|---|---|"]
for k in ("cl", "clp", "cd", "cdp"):
    t, p = col(coef, k + "_true"), col(coef, k + "_pred")
    slope = np.polyfit(t, p, 1)[0]
    lines.append(f"| {k} | {np.abs(p - t).mean():.4f} | {np.corrcoef(t, p)[0, 1]:.2f} | {slope:.2f} |")

# una frase su dove sbaglia e perché: i numeri vengono dai dati, la causa è un'ipotesi
corr_cl = np.corrcoef(cl_t, cl_p)[0, 1]
slope_cl = np.polyfit(cl_t, cl_p, 1)[0]
ratio_cd = (col(coef, "cd_pred") / col(coef, "cd_true")).mean()
worst = max(variables, key=lambda v: err("surface", v, "rmse_norm"))
lines += ["", "## Dove sbaglia", "",
          f"Il modello coglie segno e ordinamento della portanza (correlazione {corr_cl:.2f}) ma ne sottostima "
          f"l'ampiezza (pendenza {slope_cl:.2f}); la resistenza è inattendibile (Cd previsto in media {ratio_cd:.0f} volte il vero). "
          f"Sulla superficie l'errore normalizzato più alto è quello di {worst} ({err('surface', worst, 'rmse_norm'):.2f} deviazioni standard). "
          "Cause probabili, da verificare: u, v e nut a parete sono quasi nulli e la distanza dal profilo è un input, quindi sono facili; "
          "la pressione dipende invece dalla forma dell'intero profilo, che una MLP punto per punto vede solo attraverso coordinate, distanza e normale. "
          "Questo spiega anche la portanza sottostimata e il Cd, che è una differenza di piccoli contributi di pressione più una parte viscosa derivata da un campo di velocità rumoroso."]

with open("runs/results.md", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
