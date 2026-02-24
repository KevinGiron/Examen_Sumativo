#!/usr/bin/env python
"""
Script para hacer predicciones interactivas en nuevas reseñas.

Uso:
    python predict_custom.py "Mi texto a clasificar"
"""

import sys
import os
from pathlib import Path

# Asegurar que podemos importar desde src
sys.path.insert(0, str(Path(__file__).parent))

from src.predict import predict_batch
from src.embeddings import Embedder

def main():
    model_path = "models/sentiment_model_logistic.pkl"
    
    # Chequear si el modelo existe
    if not os.path.exists(model_path):
        print(f"❌ Error: Modelo no encontrado en {model_path}")
        print("   Ejecute primero: python main.py")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        # Predicción desde línea de comando
        text = " ".join(sys.argv[1:])
        make_prediction(text, model_path)
    else:
        # Modo interactivo
        interactive_mode(model_path)


def make_prediction(text, model_path):
    """Hace una predicción para un texto."""
    try:
        predictions, confidences = predict_batch([text], model_path)
        pred = predictions[0]
        conf = confidences[0]
        
        print(f"\n{'='*60}")
        print(f"PREDICCIÓN")
        print(f"{'='*60}")
        print(f"Texto: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"Clasificación: {pred.upper()}")
        print(f"Confianza: {conf:.2%}" if conf else "Confianza: N/A")
        print(f"{'='*60}\n")
        
        if conf and conf < 0.6:
            print("⚠️  Baja confianza. Revisar manualmente recomendado.")
        
    except Exception as e:
        print(f"❌ Error durante la predicción: {e}")
        sys.exit(1)


def interactive_mode(model_path):
    """Modo interactivo para múltiples predicciones."""
    print("\n" + "="*60)
    print("CLASIFICADOR DE OPINIONES - MODO INTERACTIVO")
    print("="*60)
    print("\nEscribe una reseña y presiona Enter para clasificarla.")
    print("Escribe 'salir' para terminar.\n")
    
    while True:
        try:
            text = input("Reseña: ").strip()
            
            if text.lower() in ['salir', 'exit', 'quit', 'q']:
                print("\n¡Hasta luego!")
                break
            
            if not text:
                print("⚠️  Por favor, ingresa un texto.\n")
                continue
            
            predictions, confidences = predict_batch([text], model_path)
            pred = predictions[0]
            conf = confidences[0]
            
            # Emoji según clasificación
            emoji = {'positivo': '😊', 'negativo': '😞', 'neutral': '😐'}.get(pred, '❓')
            
            result = f"{emoji} {pred.upper()}"
            if conf:
                result += f" ({conf:.0%})"
            
            print(f"Resultado: {result}\n")
            
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()
