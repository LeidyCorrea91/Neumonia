
# Detección de Neumonía con Deep Learning

## Descripción

Este proyecto utiliza redes neuronales convolucionales (CNN) para la clasificación de imágenes radiográficas en formato DICOM en tres categorías:

1. Neumonía Bacteriana
2. Neumonía Viral
3. Sin Neumonía

Se implementa la técnica Grad-CAM para resaltar las regiones clave en las imágenes usadas por la red para la clasificación.

## Requisitos

Antes de ejecutar el proyecto, asegúrese de instalar los siguientes requisitos:

```bash
pip install -r requirements.txt
```

También debe asegurarse de estar utilizando **Python 3.9**.

## Uso

### Ejecutar el proyecto en entorno local

```bash
python detector_neumonia.py
```

### Uso con Docker

```bash
docker build -t neumonia_app .
docker run -p 5000:5000 neumonia_app
```

## Arquitectura de Código

- `detector_neumonia.py`: Contiene la lógica principal del modelo.
- `integrator.py`: Integra los módulos y devuelve la clase, probabilidad y el mapa de calor.
- `read_img.py`: Carga imágenes DICOM y las convierte en arrays.
- `preprocess_img.py`: Preprocesa la imagen (redimensionado, escala de grises, ecualización).
- `load_model.py`: Carga el modelo entrenado `conv_MLP_84.h5`.
- `grad_cam.py`: Genera el mapa de calor con Grad-CAM.

## Licencia

Este proyecto se distribuye bajo la Licencia MIT.
