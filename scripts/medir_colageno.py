import numpy as np
import pandas as pd
from PIL import Image

from organoides_dl.medicion import porcentaje_dab

tabla = pd.read_excel("data/raw/recortes_colageno.xlsx")
tabla = tabla[(tabla["Usar_en_dataset"] == "Si") & (tabla["Tecnica"] == "IHQ")]

porcentajes = []
for id_imagen in tabla["ID_imagen"]:
    rgb = np.array(Image.open(f"data/raw/figuras_col/{id_imagen}.png").convert("RGB"))
    porcentajes.append(porcentaje_dab(rgb, umbral=0.025))

tabla["porcentaje"] = porcentajes
tabla[["ID_imagen", "Marcador_especifico", "Paper_DOI", "porcentaje"]].to_csv(
    "data/processed/area_colageno.csv", index=False
)
