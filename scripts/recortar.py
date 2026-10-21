from pathlib import Path

import pandas as pd
from PIL import Image

CARPETA_FIGURAS = Path("data/raw/figuras_completas")
TABLA = Path("data/raw/Recortes_Final.xlsx")
CARPETA_SALIDA = Path("data/raw/figuras")

CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
tabla = pd.read_excel(TABLA)

for fila in tabla.itertuples():
    figura = Image.open(CARPETA_FIGURAS / fila.archivo_figura)
    caja = (fila.x, fila.y, fila.x + fila.ancho, fila.y + fila.alto)
    recorte = figura.crop(caja).convert("RGB")
    recorte.save(CARPETA_SALIDA / f"{fila.ID_imagen}.png")

print(f"Listo: {len(tabla)} recortes en {CARPETA_SALIDA}")
