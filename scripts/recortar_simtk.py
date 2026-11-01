from pathlib import Path

import pandas as pd
from PIL import Image

CARPETA = Path("data/raw/Simtk")
SALIDA = Path("data/processed/simtk")
TAMANO = 600

tabla = pd.read_excel("data/raw/cuadros_simtk.xlsx")

for fila in tabla.itertuples():
    carpeta_safo = CARPETA / fila.grupo / fila.paciente / fila.muestra / "SafO"
    archivo = min(carpeta_safo.glob("*.tif"))
    imagen = Image.open(archivo).convert("RGB")
    cuadro = imagen.crop((fila.x, fila.y, fila.x + TAMANO, fila.y + TAMANO))

    destino = SALIDA / fila.grupo
    destino.mkdir(parents=True, exist_ok=True)
    cuadro.save(destino / f"{fila.paciente}_{fila.muestra}_{fila.cuadro}.png")

