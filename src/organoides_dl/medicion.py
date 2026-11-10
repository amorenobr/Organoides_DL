import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu


def porcentaje_area(imagen: Image.Image) -> float:
    hsv = np.array(imagen.convert("RGB").convert("HSV"))
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    naranja = (H <= 27) & (S >= 40) & (V >= 89)
    magenta = (H >= 198) & (S >= 85) & (V >= 89)
    tejido = (S >= 20) & (V >= 20)
    brillante = V > 215
    tenido = (naranja | magenta) & (~brillante | (S >= 100))

    return float(tenido.sum() / tejido.sum() * 100)


def porcentaje_dab(rgb, umbral=0.025):
    gris = rgb.mean(axis=2)
    tejido = binary_fill_holes(
        binary_closing(gris < threshold_otsu(gris), iterations=3)
    )
    cafe = rgb2hed(rgb)[:, :, 2] >= umbral
    return float((cafe & tejido).sum() / tejido.sum() * 100)
