from PIL import Image

from organoides_dl.preproces_densnet121 import cargar_imagen


def test_cargar_imagen_queda_en_224(tmp_path):
    ruta = tmp_path / "prueba.png"
    Image.new("RGB", (100, 50), (200, 90, 60)).save(ruta)
    imagen = cargar_imagen(ruta)
    assert imagen.shape == (224, 224, 3)
