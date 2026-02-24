import tensorflow as tf
import numpy as np
import sys
from preprocess import preprocess_image

MODEL_PATH = "../models/best_model.h5"

# Cargar modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Clases
class_names = ["apto", "defectuoso"]

def predict(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index] * 100

    print(f"Producto: {class_names[predicted_index].upper()}")
    print(f"Confianza: {confidence:.2f}%")

if __name__ == "__main__":
    image_path = sys.argv[1]
    predict(image_path)