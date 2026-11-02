from pathlib import Path

import numpy as np
from PIL import Image

for ruta in sorted(Path("data/processed/simtk").glob("*/*.png")):
    hsv = np.array(Image.open(ruta).convert("HSV"))
    tejido = ((hsv[..., 1] >= 20) & (hsv[..., 2] >= 20)).mean() * 100
    print(ruta.parent.name, ruta.stem, round(tejido, 1))
