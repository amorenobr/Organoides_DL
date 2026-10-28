from PIL import Image

from Organoides_DL.medicion import porcentaje_area


def test_porcentaje_area():
    imagen = Image.new("RGB", (100, 100), (250, 250, 250))
    imagen.paste((200, 90, 60), (0, 0, 50, 100))
    imagen.paste((120, 140, 200), (50, 0, 75, 100))

    assert round(porcentaje_area(imagen), 2) == 66.67


def test_rojo_brillante_cuenta_como_tenido():
    imagen = Image.new("RGB", (100, 100), (245, 140, 110))
    assert porcentaje_area(imagen) > 99


def test_rosado_palido_no_cuenta_como_tenido():
    imagen = Image.new("RGB", (100, 100), (250, 215, 200))
    assert porcentaje_area(imagen) < 1
