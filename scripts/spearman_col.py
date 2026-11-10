import pandas as pd
from scipy.stats import spearmanr

area = pd.read_csv("data/processed/area_colageno.csv").set_index("ID_imagen")[
    "porcentaje"
]

publicado = {
    "Col II 2D-to-3D": {
        "IMG297": 1.0,
        "IMG298": 0,
        "IMG299": 0.73,
        "IMG300": 1.58,
        "IMG301": 2.1,
    },
    "Col II Direct-3D": {
        "IMG302": 1.0,
        "IMG303": 2.55,
        "IMG304": 4.12,
        "IMG305": 2.7,
        "IMG306": 3.43,
    },
    "Col I 2D-to-3D": {
        "IMG287": 1.0,
        "IMG288": 0.65,
        "IMG289": 1.4,
        "IMG290": 1.93,
        "IMG291": 2.03,
    },
    "Col I Direct-3D": {
        "IMG292": 1.0,
        "IMG293": 1.5,
        "IMG294": 1.6,
        "IMG295": 2.02,
        "IMG296": 1.53,
    },
}

for grupo, valores in publicado.items():
    medido = [area[i] for i in valores]
    rho, p = spearmanr(medido, list(valores.values()))
    print(f"{grupo}: rho = {rho:.2f}, p = {p:.3f}")
