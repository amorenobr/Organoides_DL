import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

df = pd.read_csv("data/processed/predicciones_safranina.csv")
real = df["clase_real"]
pred = df["clase_predicha"]

print(f"Precisión balanceada: {balanced_accuracy_score(real, pred):.3f}")
print(f"Kappa ponderado: {cohen_kappa_score(real, pred, weights='quadratic'):.3f}")
print("Matriz de confusión (filas = real, columnas = predicha):")
print(confusion_matrix(real, pred))

filas = []
for ronda, g in df.groupby("fold"):
    filas.append(
        {
            "ronda": ronda,
            "precision_balanceada": balanced_accuracy_score(
                g["clase_real"], g["clase_predicha"]
            ),
            "kappa": cohen_kappa_score(
                g["clase_real"], g["clase_predicha"], weights="quadratic"
            ),
        }
    )

rondas = pd.DataFrame(filas)
print(rondas.round(2))
print(rondas[["precision_balanceada", "kappa"]].agg(["mean", "std"]).round(2))
print(classification_report(real, pred, target_names=["bajo", "medio", "alto"]))
