from integrator import predict_image
from tkinter import filedialog
from PIL import ImageTk, Image

class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Detector de Neumonía")

        # Botón para cargar imagen
        self.button_load = ttk.Button(self.root, text="Cargar Imagen", command=self.load_img_file)
        self.button_load.pack()

        # Botón para predecir
        self.button_predict = ttk.Button(self.root, text="Predecir", state="disabled", command=self.run_model)
        self.button_predict.pack()

        self.image_path = None
        self.root.mainloop()

    def load_img_file(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("DICOM Files", "*.dcm")])
        if self.image_path:
            self.button_predict["state"] = "enabled"

    def run_model(self):
        label, prob, heatmap = predict_image(self.image_path)
        print(f"Predicción: {label} con {prob:.2f}% de certeza")

if __name__ == "__main__":
    app = App()
