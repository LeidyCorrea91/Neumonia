import cv2
import os

def read_image(file_path):
    """Carga una imagen JPG en escala de grises y verifica si el archivo existe."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo no existe: {file_path}")

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {file_path}")

    return image
