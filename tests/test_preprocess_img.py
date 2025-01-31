import pytest
import numpy as np
from src.data.preprocess_img import preprocess_image

def test_preprocess_image():
    """Prueba que el preprocesamiento de la imagen se haga correctamente."""
    sample_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)  # Imagen simulada
    processed_img = preprocess_image(sample_image)
    
    # Verificar que el resultado tenga las dimensiones correctas
    assert processed_img.shape == (1, 512, 512, 1), "El preprocesamiento no tiene la forma esperada"

    # Verificar que los valores están normalizados entre 0 y 1
    assert 0.0 <= processed_img.min() and processed_img.max() <= 1.0, "La normalización falló"
