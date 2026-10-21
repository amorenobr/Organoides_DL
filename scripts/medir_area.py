from pathlib import Path

import pandas as pd
from PIL import Image

from Organoides_DL.medicion import porcentaje_area

CARPETA_IMAGENES = Path("data/raw/figuras")
SALIDA = Path("data/processed/area_safranina.csv")

resultados = []

for ruta in sorted(CARPETA_IMAGENES.glob("*.png")):
    porcentaje = porcentaje_area(Image.open(ruta))
    resultados.append({"ID_imagen": ruta.stem, "porcentaje_area": round(porcentaje, 2)})

pd.DataFrame(resultados).to_csv(SALIDA, index=False)

