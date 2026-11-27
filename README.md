# Proyecto Modelo de Clasificación de Imágenes usando DL

Clasificación de la intensidad de tinción en imágenes de organoides de cartílago con redes neuronales convolucionales (CNN)

Trabajo de grado para la carrera de Biotecnología, Universidad Antonio Nariño

## Objetivo
Clasificar imágenes de organoides de cartílago en tres niveles de tinción (bajo < 10%, medio 10-70 %, alto > 70% de área teñida) para safranina-O e inmunotinción de colágeno II,
y compararlas con tejido articular humano sano y degenerado.

## Datos
- Imágenes recortadas de figuras de artículos publicados (lista de DOI en `data/raw/dataset_v1_2026_10_05.xlsx`). No se incluyen en el repositorio por derechos de autor.
- Tejido articular humano.
- Safranina-O: 123 imágenes de 16 artículos.
- Colágeno II: 75 imágenes de 13 artículos.

## Método
1. Recorte de las imágenes por coordenadas (`recortar.py`, `recortar_col.py`)
2. Medición del porcentaje de área teñida con umbrales de color (`pixi run medir`, `pixi run medir_col`)
3. División en 5 grupos por artículo, para que ningún artículo esté a la vez en entrenamiento y en prueba
(`pixi run preparar_safo`, `pixi run preparar_colII`)
4. Entrenamiento de DenseNet121 preentrenada en ImageNet con una capa de salida nueva (`pixi run entrenar`, `pixi run entrenar_colII`)
5. Métricas, pruebas de permutación, y Grad-CAM (`pixi run metricas`,`pixi run metricas_colII`)
6. Índice de similitud con tejido sano/degenerado (`similitud.py`)

## Resultados
| Tinción | Precisión balanceada | Kappa ponderado | Clase mayoritaria |
|---------|----------------------|-----------------|-------------------|
| Safranina-O | 0.72 | 0.73 | 0.43 |
| Colágeno II | 0.65 | 0.66 | 0.49 |

Validación cruzada agrupada por artículo (5 grupos).
"Clase mayoritaria" es la precisión que se obtiene prediciendo siempre la clase más frecuente.

## Limitaciones
- Las clases están definidas a partir de umbrales de color, no por evaluación experta. El modelo aprende a reproducir esa regla.
- Pocas imágenes (123 y 75) y pocos artículos (16 y 13); 8 de los 16 artículos de safranina tienen una sola clase.

## Instalación y uso
```bash 
pixi install
pixi run test
```

## Estructura
```
src/organoides_dl/      funciones (medición, modelo, preprocesamiento)
scripts/                pasos del análisis
tests/                  pruebas
data/processed/         resultados intermedios (csv)
reports/                figuras
```

## Licencia
El código se distribuye bajo la licencia MIT (ver `LICENSE`). Las imágenes provienen de artículos publicados y conservan los derechos
de sus autores y editoriales.

## Autor
Juan Felipe Coy Carrillo, Biotecnología, Universidad Antonio Nariño.
Director: Alexander Moreno Briceño.
