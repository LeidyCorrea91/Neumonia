from read_img import read_dicom_file
from preprocess_img import preprocess_image
from load_model import load_trained_model
from grad_cam import generate_grad_cam
import numpy as np

def predict_image(image_path):
    """Realiza la predicción sobre la imagen preprocesada y genera el Grad-CAM."""
    img_rgb = read_dicom_file(image_path)
    preprocessed_img = preprocess_image(img_rgb)
    
    model = load_trained_model()
    predictions = model.predict(preprocessed_img)
    predicted_class = np.argmax(predictions)
    probability = np.max(predictions) * 100
    labels = ["Bacteriana", "Normal", "Viral"]

    grad_cam = generate_grad_cam(preprocessed_img, model)
    
    return labels[predicted_class], probability, grad_cam

