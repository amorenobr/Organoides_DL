import sys

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras

from organoides_dl.medicion import porcentaje_area, porcentaje_dab
from organoides_dl.preproces_densnet121 import cargar_imagen

tincion, ruta = sys.argv[1], sys.argv[2]
imagen = cargar_imagen(ruta)[None]

cnn = keras.models.load_model(f"modelos/{tincion}.keras")
clase = ["bajo", "medio", "alto"][cnn.predict(imagen).argmax()]

if tincion == "safranina":
    area = porcentaje_area(Image.open(ruta))
else:
    area = porcentaje_dab(np.array(Image.open(ruta).convert("RGB")))

print(f"% de área teñida: {area:.1f}")
print(f"Clase según la CNN: {clase}")

if tincion == "safranina":
    huellas = pd.read_csv("data/processed/huellas.csv")
    numeros = huellas.columns[2:]
    huellas["paciente"] = huellas["imagen"].str.split("_").str[0]
    pacientes = huellas[huellas["grupo"] != "figuras"].groupby(["grupo", "paciente"])[numeros].mean()
    degenerado = pacientes.loc["degenerado"].mean().to_numpy()
    eje = pacientes.loc["sano"].mean().to_numpy() - degenerado

    densenet = keras.applications.DenseNet121(weights="imagenet", include_top=False, pooling="avg")
    huella = densenet.predict(keras.applications.densenet.preprocess_input(imagen))[0]
    indice = (huella - degenerado) @ eje / (eje @ eje)
    atipicidad = np.linalg.norm((huella - degenerado) - indice * eje) / np.linalg.norm(eje)

    print(f"Índice de similitud: {indice:.2f}  (0 = degenerado, 1 = sano)")
    print(f"Atipicidad: {atipicidad:.2f}  (tejido nativo ≈ 1.4)")