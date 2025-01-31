import pytest
from src.data.integrator import predict_image

@pytest.mark.parametrize("image_path", [
    "tests/sample_images/sample1.dcm", 
    "tests/sample_images/sample2.dcm"
])
def test_predict_image(image_path):
    """Prueba que la predicción del modelo devuelva una clase válida y probabilidad numérica."""
    label, probability, heatmap = predict_image(image_path)

    # Verificar que la etiqueta de salida sea una de las esperadas
    expected_labels = ["Bacteriana", "Normal", "Viral"]
    assert label in expected_labels, f"La etiqueta '{label}' no está en {expected_labels}"

    # Verificar que la probabilidad sea un número entre 0 y 100
    assert 0.0 <= probability <= 100.0, f"La probabilidad '{probability}' está fuera de rango"

    # Verificar que el heatmap tenga las dimensiones esperadas
    assert heatmap is not None, "El heatmap no fue generado correctamente"
