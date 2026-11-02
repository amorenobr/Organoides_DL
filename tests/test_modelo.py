import numpy as np

from organoides_dl.modelo import construir_modelo


def test_modelo_probabilidades():
    modelo = construir_modelo()
    salida = modelo.predict(np.zeros((1, 224, 224, 3)))
    assert salida.shape == (1, 3)
    assert abs(salida.sum() - 1) < 1e-5
