import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

area = pd.read_csv("data/processed/area_safranina.csv")
dataset = pd.read_excel("data/raw/dataset_v1_2026_10_05.xlsx")
columnas = ["ID_imagen", "Paper_DOI", "Usar_en_dataset"]

union = pd.merge(area, dataset[columnas], on="ID_imagen", how="left")


print(union["Paper_DOI"].isna().sum())

union = union[union["Usar_en_dataset"] == "Si"].reset_index(drop=True)

union["clase"] = pd.cut(
    union["porcentaje_area"],
    bins=[0, 10, 70, 100.01],
    labels=["bajo", "medio", "alto"],
    include_lowest=True,
    right=False,
)
doi_yamashita = "10.1016/j.stemcr.2015.01.016"
es_yamashita = union["Paper_DOI"] == doi_yamashita
print(es_yamashita.sum())

yamashita = union[es_yamashita]
seleccion = pd.concat(
    [
        yamashita[yamashita["clase"] == "medio"],
        yamashita[yamashita["clase"] == "alto"].sample(frac=0.5, random_state=42),
        yamashita[yamashita["clase"] == "bajo"].sample(frac=0.5, random_state=42),
    ]
)
union = pd.concat([union[~es_yamashita], seleccion]).reset_index(drop=True)

print(union["clase"].value_counts())

divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
union["fold"] = -1

for ronda, (_, indices_prueba) in enumerate(
    divisor.split(union, union["clase"], groups=union["Paper_DOI"])
):
    union.loc[indices_prueba, "fold"] = ronda

print(pd.crosstab(union["fold"], union["clase"]))

union.to_csv("data/processed/safranina_folds.csv", index=False)
