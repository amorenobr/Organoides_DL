import pandas as pd
from scipy.stats import spearmanr

area = pd.read_csv("data/processed/area_safranina.csv").set_index("ID_imagen")[
    "porcentaje_area"
]

publicado = {
    "Chen": {
        "IMG415": 0,
        "IMG416": 5,
        "IMG417": 10,
        "IMG418": 20,
        "IMG419": 2,
        "IMG420": 1,
    },
    "Peng": {
        "IMG522": 0.66,
        "IMG523": 0.44,
        "IMG524": 0.17,
        "IMG525": 0.21,
        "IMG526": 0.14,
        "IMG527": 0.11,
    },
    "Koenig_dias": {
        "IMG110": 8,
        "IMG393": 18,
        "IMG394": 35,
        "IMG395": 50,
        "IMG396": 72,
        "IMG397": 88,
    },
    "Koenig_AHA": {
        "IMG531": 25,
        "IMG532": 23,
        "IMG533": 9,
        "IMG534": 2.5,
        "IMG535": 2.5,
    },
}

for grupo, valores in publicado.items():
    medido = [area[i] for i in valores]
    rho, p = spearmanr(medido, list(valores.values()))
    print(f"{grupo}: rho = {rho:.2f}, p = {p:.3f}")
