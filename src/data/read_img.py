import pydicom as dicom
import cv2
import numpy as np

def read_dicom_file(path):
    """Carga una imagen en formato DICOM y la convierte a RGB."""
    img = dicom.dcmread(path)
    img_array = img.pixel_array
    img_scaled = (np.maximum(img_array, 0) / img_array.max()) * 255.0
    img_scaled = np.uint8(img_scaled)
    img_rgb = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2RGB)
    return img_rgb

