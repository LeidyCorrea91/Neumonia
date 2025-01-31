import numpy as np
import cv2
import tensorflow as tf
import pydicom as dicom
from PIL import Image
from keras.models import load_model

# Deshabilitar eager execution para compatibilidad con versiones antiguas
tf.compat.v1.disable_eager_execution()

def read_dicom_file(path):
    """Carga una imagen en formato DICOM y la convierte a RGB."""
    img = dicom.dcmread(path)
    img_array = img.pixel_array
    img_scaled = (np.maximum(img_array, 0) / img_array.max()) * 255.0
    img_scaled = np.uint8(img_scaled)
    img_rgb = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2RGB)
    return img_rgb

def preprocess_image(array):
    """Preprocesa la imagen para el modelo de predicción."""
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255.0
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array

def load_trained_model():
    """Carga el modelo previamente entrenado."""
    return load_model('conv_MLP_84.h5')

def predict_image(array):
    """Realiza la predicción sobre la imagen preprocesada."""
    model = load_trained_model()
    preprocessed_img = preprocess_image(array)
    predictions = model.predict(preprocessed_img)
    predicted_class = np.argmax(predictions)
    probability = np.max(predictions) * 100
    labels = ["Bacteriana", "Normal", "Viral"]
    return labels[predicted_class], probability
