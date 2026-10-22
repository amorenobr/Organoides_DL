import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

area = pd.read_csv("data/processed/area_safranina.csv")
dataset = pd.read_excel("data/raw/dataset_v1_2026_10_05.xlsx")
columnas = ["ID_imagen", "Paper_DOI"]

union = pd.merge(area, dataset[columnas], on="ID_imagen", how="left")

print(union["Paper_DOI"].isna().sum())

union["clase"] = pd.cut(
    union["porcentaje_area"],
    bins=[0, 10, 50, 100.01],
    labels=["bajo", "medio", "alto"],
    include_lowest=True,
    right=False,
)

print(union["clase"].value_counts())

divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
union["fold"] = -1

for ronda, (_, indices_prueba) in enumerate(
    divisor.split(union, union["clase"], groups=union["Paper_DOI"])
):
    union.loc[indices_prueba, "fold"] = ronda

print(pd.crosstab(union["fold"], union["clase"]))

union.to_csv("data/processed/safranina_folds.csv", index=False)