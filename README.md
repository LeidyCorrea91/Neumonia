
# 📌 Detector de Neumonía con Inteligencia Artificial

## 📝 Descripción del Proyecto
Este proyecto implementa un modelo de inteligencia artificial para la detección de neumonía a partir de imágenes DICOM de rayos X e imagenes JPG. Se utiliza una **red neuronal convolucional (CNN)** preentrenada  para la clasificación de imágenes radiográficas en formato DICOM en tres categorías 

1. Neumonía Bacteriana
2. Neumonía Viral
3. Sin Neumonía
  
Generando predicciones y mapas de calor mediante **Grad-CAM**.

Este repositorio incluye el código necesario para procesar imágenes médicas, cargar el modelo y desplegar la aplicación utilizando **Docker**.

---
## 📁 Estructura del Proyecto

La estructura del código está organizada de la siguiente manera:

```
Neumonia-Detector/
│── data/                     # Carpeta para almacenar datos
│   ├── JPG/                  # Muestra de imagenes
│
│
│── src/                      # Código fuente
│   ├── data/
│   │   ├── read_img.py       # Carga imágenes DICOM y JPG
│   │   ├── preprocess_img.py # Preprocesa imágenes (normalización, escalado, etc.)
│   │
│   ├── models/
│   │   ├── load_model.py     # Carga el modelo de red neuronal
│   │
│   ├── utils/
│   │   ├── grad_cam.py       # Genera mapas de calor con Grad-CAM
│   │   ├── integrator.py     # Une los procesos de predicción y visualización
│
│── tests/                    # Pruebas unitarias con pytest
│   ├── test_data_processing.py
│   ├── test_model.py
│
│── docker/                   # Archivos de configuración para Docker
│   ├── Dockerfile
│   ├── .dockerignore
│
│── README.md                 # Documentación del proyecto
│── requirements.txt           # Librerías necesarias
│── .gitignore                 # Archivos ignorados en Git
```

---

## 📌 Prerrequisitos

Antes de ejecutar el proyecto, asegúrate de tener instalado lo siguiente:

### **🔹 Requisitos del Sistema**
- **Sistema Operativo:** Windows, macOS o Linux
- **Python:** 3.9
- **Docker:** Última versión estable

### **🔹 Instalación de Dependencias**
Si deseas ejecutar el código sin Docker, instala las dependencias manualmente con:

```bash
pip install -r requirements.txt
```

---
## 🐳 Configuración y Ejecución con Docker

Para ejecutar el sistema en un entorno aislado, utilizamos **Docker**.

### **1️⃣ Construcción de la Imagen**
Ejecuta el siguiente comando en la terminal dentro del directorio del proyecto:
```bash
docker build -t detector-neumonia .
```

### **2️⃣ Verificar la Instalación de Dependencias en Docker**
Para asegurarte de que `pydicom` y `opencv-python` están correctamente instalados dentro del contenedor, usa:
```bash
docker run --rm detector-neumonia:jg pip list | grep -E "pydicom|opencv-python"
```

### **3️⃣ Ejecutar el Contenedor con una Imagen de Entrada**
```bash
docker run --rm -v "C:\Users\DELL\Downloads\Neumonia-main\Neumonia-main\JPG:/app/JPG" detector-neumonia --file "/app/JPG/person1710_bacteria_4526.jpeg"
```

### **4️⃣ Ingresar al Contenedor Interactivo**
Si necesitas depurar dentro del contenedor:
```bash
docker run --rm -it detector-neumonia:jg /bin/sh
```

### **5️⃣ Verificar la Salida del Modelo**
Para ejecutar el modelo manualmente dentro del contenedor:
```bash
docker run --rm detector-neumonia:jg python src/data/detector_neumonia.py
```

---

## 📊 Resultados Esperados

### 🔹 **Salida de Predicción**
La salida del modelo contiene:
- **Clase Predicha:** Neumonía / Normal
- **Probabilidad:** Confianza del modelo en la predicción
- **Mapa de Calor:** Generado con Grad-CAM para visualizar áreas de interés

Ejemplo de salida:
```json
{
    "prediction": "Neumonía",
    "probability": 0.97,
    "heatmap": "gradcam_output.png"
}
```

---

## 🔧 Solución de Problemas

### **🔹 Error `ModuleNotFoundError: No module named 'pydicom'`**
📌 Solución:
```bash
docker run --rm detector-neumonia:jg pip install pydicom
```

### **🔹 Error `ImportError: libGL.so.1: cannot open shared object file`**
📌 Solución:
```bash
docker build --no-cache -t detector-neumonia:jg .
```
Si persiste, accede al contenedor e instala manualmente:
```bash
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
```


## Arquitectura de Código

- `detector_neumonia.py`: Contiene la lógica principal del modelo.
- `integrator.py`: Integra los módulos y devuelve la clase, probabilidad y el mapa de calor.
- `read_img.py`: Carga imágenes DICOM y las convierte en arrays.
- `preprocess_img.py`: Preprocesa la imagen (redimensionado, escala de grises, ecualización).
- `load_model.py`: Carga el modelo entrenado `conv_MLP_84.h5`.
- `grad_cam.py`: Genera el mapa de calor con Grad-CAM.
---

## 📜 Licencia

Este proyecto está bajo la licencia **MIT**. Puedes ver más detalles en el archivo `LICENSE`.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
```

---
## 📩 Contacto
Si tienes alguna pregunta, contáctame en:
📧 Email: example@email.com
📌 GitHub: [https://github.com/tuusuario](https://github.com/tuusuario)

---
🚀 **¡Listo para ejecutar el detector de neumonía con IA!** 🎯

