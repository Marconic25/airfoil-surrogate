import csv
import matplotlib.pyplot as plt

epochs, train, val = [], [], []
with open("runs/log.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train.append(float(row["train_loss"]))
        val.append(float(row["val_loss"]))

plt.plot(epochs, train, label="train")
plt.plot(epochs, val, label="validation")
plt.yscale("log")                                    # scala log: si vedono bene valori tra 0.01 e 1
plt.xlabel("epoca")
plt.ylabel("MSE (target normalizzati)")
plt.legend()
plt.savefig("runs/curves.png", dpi=150)              # savefig e non show(): in WSL non c'è una finestra grafica, scrive il file e lo apri dall'editor
