import cv2
import numpy as np
from tensorflow.keras.models import load_model
from read_img import read_image

# Cargar el modelo
modelo = load_model("/app/conv_MLP_84.h5")

def predict_image(image_path):
    """Carga una imagen JPG y la usa para predecir neumonía con el modelo."""
    image = read_image(image_path)  # Cargar imagen en escala de grises
    image = cv2.resize(image, (512, 512))  # Ajustar al tamaño requerido por el modelo
    image = image.astype(np.float32) / 255.0  # Normalizar la imagen

    # Expandir dimensiones para que coincida con la entrada del modelo (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)  # Agregar canal si el modelo lo requiere

    # Hacer la predicción con el modelo
    prediction = modelo.predict(image)

    # Extraer el primer valor del array de predicciones
    pred_value = prediction[0][0]  # Obtener el primer elemento

    label = "Neumonía" if pred_value > 0.5 else "Normal"
    prob = pred_value * 100  # Convertir probabilidad a porcentaje

    return label, prob
