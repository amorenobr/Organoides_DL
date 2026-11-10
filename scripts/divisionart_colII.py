import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

tabla = pd.read_csv("data/processed/area_colageno.csv")
tabla = tabla[tabla["Marcador_especifico"] == "Col_II"].reset_index(drop=True)

tabla["clase"] = pd.cut(
    tabla["porcentaje"],
    [0, 10, 70, 100.01],
    labels=["bajo", "medio", "alto"],
    include_lowest=True,
).astype(str)

tabla["fold"] = -1
division = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (_, prueba) in enumerate(
    division.split(tabla, tabla["clase"], tabla["Paper_DOI"])
):
    tabla.loc[prueba, "fold"] = fold

tabla.to_csv("data/processed/colageno_folds.csv", index=False)
print(tabla.groupby("fold")["clase"].value_counts().unstack(fill_value=0))
