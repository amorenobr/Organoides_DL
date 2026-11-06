import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu

for id_imagen in ["IMG209", "IMG193"]: 
    rgb = np.array(Image.open(f"data/raw/figuras_col/{id_imagen}.png").convert("RGB"))
    dab = rgb2hed(rgb)[:, :, 2]
    gris = rgb.mean(axis=2)
    tejido = gris < threshold_otsu(gris)
   

    tejido = binary_fill_holes(binary_closing(tejido, iterations=3))
    print(id_imagen, tejido.mean().round(2), np.percentile(dab[tejido], [5, 25, 50, 75, 95]).round(3))

    Image.fromarray(tejido).save(f"mascara_{id_imagen}.png")
