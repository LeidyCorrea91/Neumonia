from keras.models import load_model

def load_trained_model():
    """Carga el modelo previamente entrenado."""
    return load_model('models/conv_MLP_84.h5')

