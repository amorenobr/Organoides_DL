import numpy as np

from organoides_dl.medicion import porcentaje_dab


def organoide(color):
    imagen = np.full((100, 100, 3), 230, dtype=np.uint8)
    y, x = np.ogrid[:100, :100]
    imagen[(y - 50) ** 2 + (x - 50) ** 2 < 30**2] = color
    return imagen


def test_dab_cafe():
    assert porcentaje_dab(organoide((150, 110, 70))) > 99


def test_dab_sin_cafe():
    assert porcentaje_dab(organoide((200, 200, 205))) < 1
