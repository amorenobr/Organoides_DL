import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from organoides_dl.preproces_densnet121 import cargar_imagen

modelo = keras.models.load_model("modelos/safranina.keras")
base = modelo.get_layer("densenet121")
salida = modelo.layers[-1]

for id_imagen in sys.argv[1:]:
    imagen = cargar_imagen(f"data/raw/figuras/{id_imagen}.png")[None]
    entrada = keras.applications.densenet.preprocess_input(imagen.copy())

    with tf.GradientTape() as cinta:
        mapas = base(entrada, training=False)
        probs = salida(tf.reduce_mean(mapas, axis=(1, 2)))
        clase = int(tf.argmax(probs[0]))
        puntaje = probs[:, clase]

    gradientes = cinta.gradient(puntaje, mapas)
    pesos = tf.reduce_mean(gradientes, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(mapas[0] * pesos[0], axis=-1)).numpy()
    cam = tf.image.resize(cam[..., None] / (cam.max() + 1e-8), (224, 224)).numpy()[..., 0]

    plt.figure()
    plt.imshow(imagen[0].astype("uint8"))
    plt.imshow(cam, cmap="jet", alpha=0.4)
    plt.title(f"{id_imagen}: {['bajo', 'medio', 'alto'][clase]}")
    plt.axis("off")
    plt.savefig(f"reports/gradcam_{id_imagen}.png", dpi=200, bbox_inches="tight")