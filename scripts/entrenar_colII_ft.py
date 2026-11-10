from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from organoides_dl.modelo import construir_modelo, descongelar
from organoides_dl.preproces_densnet121 import cargar_imagen

keras.utils.set_random_seed(42)

CARPETA = Path("data/raw/figuras_col")
tabla = pd.read_csv("data/processed/colageno_folds.csv")
clases = {"bajo": 0, "medio": 1, "alto": 2}

X = np.stack([cargar_imagen(CARPETA / f"{i}.png") for i in tabla["ID_imagen"]])
y = tabla["clase"].map(clases).to_numpy()

resultados = []
curvas = []

for ronda in range(5):
    prueba = (tabla["fold"] == ronda).to_numpy()
    entrenamiento = ~prueba

    pesos = compute_class_weight(
        "balanced", classes=np.unique(y[entrenamiento]), y=y[entrenamiento]
    )

    modelo = construir_modelo()
    historia = modelo.fit(
        X[entrenamiento],
        y[entrenamiento],
        validation_data=(X[prueba], y[prueba]),
        epochs=30,
        batch_size=32,
        class_weight=dict(enumerate(pesos)),
    )
    for epoca, (perdida, perdida_val) in enumerate(
        zip(historia.history["loss"], historia.history["val_loss"]), start=1
    ):
        curvas.append(
            {
                "fold": ronda,
                "epoca": epoca,
                "entrenamiento": perdida,
                "validacion": perdida_val,
            }
        )

    descongelar(modelo)
    historia_ft = modelo.fit(
        X[entrenamiento],
        y[entrenamiento],
        validation_data=(X[prueba], y[prueba]),
        epochs=10,
        batch_size=32,
        class_weight=dict(enumerate(pesos)),
    )
    for epoca, (perdida, perdida_val) in enumerate(
        zip(historia_ft.history["loss"], historia_ft.history["val_loss"]), start=31
    ):
        curvas.append(
            {
                "fold": ronda,
                "epoca": epoca,
                "entrenamiento": perdida,
                "validacion": perdida_val,
            }
        )
    probs = modelo.predict(X[prueba])
    for id_img, real, p in zip(tabla["ID_imagen"][prueba], y[prueba], probs):
        resultados.append(
            {
                "ID_imagen": id_img,
                "fold": ronda,
                "clase_real": real,
                "clase_predicha": int(p.argmax()),
                "p_bajo": p[0],
                "p_medio": p[1],
                "p_alto": p[2],
            }
        )

pd.DataFrame(curvas).to_csv("data/processed/curvas_colII_ft.csv", index=False)
pd.DataFrame(resultados).to_csv(
    "data/processed/predicciones_colII_ft.csv", index=False
)
