import argparse
import cv2
import numpy as np
from integrator import predict_image  


def main():
    # Parser de argumentos
    parser = argparse.ArgumentParser(description="Detector de Neumonía en imágenes JPG")
    parser.add_argument("--file", required=True, help="Ruta de la imagen JPG a analizar")
    args = parser.parse_args()

    print(f"Procesando la imagen: {args.file}")

    try:
        # Ejecutar la predicción con la imagen JPG
        label, prob = predict_image(args.file)
        print(f"Predicción: {label} con {prob:.2f}% de certeza")

    except Exception as e:
        print(f"Error al procesar la imagen: {e}")

if __name__ == "__main__":
    main()

