import numpy as np
from PIL import Image


def porcentaje_area(imagen: Image.Image) -> float:
    hsv = np.array(imagen.convert("RGB").convert("HSV"))
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    naranja = (H <= 27) & (S >= 40) & (V >= 89) & (V <= 215)
    magenta = (H >= 198) & (S >= 85) & (V >= 89) & (V <= 215)
    tejido = (S >= 20) & (V >= 20)
    tenido = naranja | magenta

    return float(tenido.sum() / tejido.sum() * 100)
