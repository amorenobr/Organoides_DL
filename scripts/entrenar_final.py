from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from organoides_dl.modelo import construir_modelo
from organoides_dl.preproces_densnet121 import cargar_imagen

keras.utils.set_random_seed(42)
clases = {"bajo": 0, "medio": 1, "alto": 2}
Path("modelos").mkdir(exist_ok=True)

for carpeta, folds, nombre in [
    ("data/raw/figuras", "data/processed/safranina_folds.csv", "safranina"),
    ("data/raw/figuras_col", "data/processed/colageno_folds.csv", "colagenoII"),
]:
    tabla = pd.read_csv(folds)
    X = np.stack(
        [cargar_imagen(Path(carpeta) / f"{i}.png") for i in tabla["ID_imagen"]]
    )
    y = tabla["clase"].map(clases).to_numpy()
    pesos = compute_class_weight("balanced", classes=np.unique(y), y=y)

    modelo = construir_modelo()
    modelo.fit(X, y, epochs=30, batch_size=32, class_weight=dict(enumerate(pesos)))
    modelo.save(f"modelos/{nombre}.keras")
