from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from Organoides_DL.augmentation import augmentation
from Organoides_DL.medicion import porcentaje_area

CARPETA = Path("data/raw/figuras")
tabla = pd.read_csv("data/processed/safranina_folds.csv")
aumento = augmentation()
filas = []

for fila in tabla.itertuples():
    img = np.array(Image.open(CARPETA / f"{fila.ID_imagen}.png").convert("RGB"))
    img = tf.image.resize_with_pad(img, 224, 224).numpy()

    for _ in range(10):
        nueva = aumento(img[None], training=True)[0].numpy()
        nueva = np.clip(nueva, 0, 255).astype("uint8")
        filas.append({"ID_imagen": fila.ID_imagen,
                      "pct_original": fila.porcentaje_area,
                      "pct_aumentada": porcentaje_area(Image.fromarray(nueva))})

df = pd.DataFrame(filas)
cortes = [0, 10, 70, 100.01]
df["clase_original"] = pd.cut(df["pct_original"], cortes, right=False)
df["clase_aumentada"] = pd.cut(df["pct_aumentada"], cortes, right=False)
df.to_csv("data/processed/verificacion_aumento.csv", index=False)

cambio = (df["clase_original"] != df["clase_aumentada"]).mean() * 100
print(f"Cambio de clase: {cambio:.1f}%")
