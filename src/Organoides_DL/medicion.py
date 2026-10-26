import numpy as np
from PIL import Image


def porcentaje_area(imagen: Image.Image) -> float:
    hsv = np.array(imagen.convert("RGB").convert("HSV"))
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    naranja = (H <= 27) & (S >= 40) & (V >= 89) 
    magenta = (H >= 198) & (S >= 85) & (V >= 89) 
    tejido = (S >= 20) & (V >= 20)
    brillante = V > 215
    tenido = (naranja | magenta) & (~brillante | (S >= 100))

    return float(tenido.sum() / tejido.sum() * 100)
