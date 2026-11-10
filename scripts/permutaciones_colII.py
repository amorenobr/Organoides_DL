import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

pred = pd.read_csv("data/processed/predicciones_colagenoII.csv")
real, predicha = pred["clase_real"], pred["clase_predicha"]

kappa_real = cohen_kappa_score(real, predicha, weights="quadratic")

rng = np.random.default_rng(42)
kappas_azar = [
    cohen_kappa_score(rng.permutation(real), predicha, weights="quadratic")
    for _ in range(10000)
]

veces = sum(k >= kappa_real for k in kappas_azar)
p = (veces + 1) / (10000 + 1)
print(
    f"kappa real = {kappa_real:.3f}, kappa azar promedio = {np.mean(kappas_azar):.3f}, p = {p:.4f}"
)

kappas_boot = []
for _ in range(10000):
    i = rng.integers(0, len(real), len(real))
    kappas_boot.append(
        cohen_kappa_score(real.iloc[i], predicha.iloc[i], weights="quadratic")
    )
print(
    f"IC 95 % del kappa: {np.percentile(kappas_boot, 2.5):.2f} a {np.percentile(kappas_boot, 97.5):.2f}"
)
