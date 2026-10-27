import numpy as np
from PIL import Image, ImageOps


def cargar_imagen(ruta):
    imagen = Image.open(ruta).convert("RGB")
    imagen = ImageOps.pad(imagen, (224, 224), color=(255, 255, 255))
    return np.array(imagen, dtype="float32")