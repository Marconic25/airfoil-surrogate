-non so come strutturare uno script da zero, soprattutto split e statistiche
- non ricordo come fare tutte queste cose
Serve un generatore con seed: np.random.default_rng(seed).
Il generatore ha un metodo .permutation(n) che ti dà gli indici mescolati.
Calcola n_train = int(0.7 * n) e n_val = int(0.15 * n). Poi taglia l’array con le slice: [:n_train], [n_train:n_train+n_val], [n_train+n_val:].
Alla fine fai xs[idx_train], xs[idx_val] e xs[idx_test], e lo stesso per ys.

- non so scrivere funzione di permutazione

- non so cosa siano tensori float32.

-n_train = 0.7 * perm moltiplica l’array intero. Doveva essere int(0.7 * n): un numero, non un array. Lo stesso per n_val.
perm ha una sola dimensione, quindi perm[:, 0:n_train] è sbagliato. Si scrive perm[:n_train].
Il val non è perm[:, n_train:n_val]: finisce a n_train + n_val, non a n_val.
Il test non è perm[:, n_val:-1]: parte da n_train + n_val e va fino alla fine, perm[n_train + n_val:]. Il -1 ti avrebbe tolto l’ultimo elemento.

L’idea era giusta (default_rng, permutation, tre slice). Gli errori sono tutti di indicizzazione.