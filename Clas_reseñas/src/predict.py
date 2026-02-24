import joblib
import numpy as np
from .embeddings import Embedder

def predict(text, model_path="models/sentiment_model.pkl"):
    """
    Predice la clase de un texto.
    
    Args:
        text: Texto a clasificar
        model_path: Ruta al modelo entrenado
        
    Returns:
        prediction: Etiqueta predicha
        probability: Probabilidad o score de predicción
    """
    model = joblib.load(model_path)
    embedder = Embedder()
    
    embedding = embedder.encode([text])
    
    prediction = model.predict(embedding)[0]
    
    # Intentar obtener probabilidades si el modelo las soporta
    try:
        proba = model.predict_proba(embedding)[0]
        probability = float(np.max(proba))
    except:
        probability = None
    
    return prediction, probability


def predict_batch(texts, model_path="models/sentiment_model.pkl"):
    """
    Predice las clases para múltiples textos.
    
    Args:
        texts: Lista de textos
        model_path: Ruta al modelo entrenado
        
    Returns:
        predictions: Lista de predicciones
        probabilities: Lista de probabilidades
    """
    model = joblib.load(model_path)
    embedder = Embedder()
    
    embeddings = embedder.encode(texts)
    predictions = model.predict(embeddings)
    
    probabilities = []
    try:
        probas = model.predict_proba(embeddings)
        probabilities = [float(np.max(p)) for p in probas]
    except:
        probabilities = [None] * len(predictions)
    
    return predictions, probabilities