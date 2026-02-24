# Clasificador de Productos - Apto / Defectuoso

Modelo: MobileNetV2 (Transfer Learning)
Framework: TensorFlow / Keras

## Estructura

- dataset/: imágenes organizadas por clase
- src/train.py: entrenamiento
- src/predict.py: predicción
- models/: modelo entrenado

## Cómo entrenar

python src/train.py

## Cómo predecir

python src/predict.py ruta_imagen.jpg