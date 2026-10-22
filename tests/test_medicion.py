from PIL import Image

from Organoides_DL.medicion import porcentaje_area


def test_porcentaje_area():
    imagen = Image.new("RGB", (100, 100), (250, 250, 250))
    imagen.paste((200, 90, 60), (0, 0, 50, 100))
    imagen.paste((120, 140, 200), (50, 0, 75, 100))

    assert round(porcentaje_area(imagen), 2) == 66.67
