#!/usr/bin/env python
"""
Script de análisis detallado del dataset y resultados del modelo.

Genera estadísticas del dataset y análisis de performance.
"""

import pandas as pd
import os
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_csv_data
from src.embeddings import Embedder

def analyze_dataset(csv_path="data/raw/reviews.csv"):
    """Analiza estadísticas del dataset."""
    
    if not os.path.exists(csv_path):
        print(f"❌ Dataset no encontrado: {csv_path}")
        return
    
    # Cargar datos
    texts, labels = load_csv_data(csv_path)
    
    print("\n" + "="*70)
    print("ANÁLISIS DEL DATASET")
    print("="*70)
    
    # Estadísticas generales
    print(f"\n📊 Estadísticas Generales:")
    print(f"  Total de reseñas: {len(texts)}")
    print(f"  Número de clases: {len(set(labels))}")
    print(f"  Clases: {', '.join(sorted(set(labels)))}")
    
    # Distribución de clases
    print(f"\n📈 Distribución de Clases:")
    class_counts = Counter(labels)
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / len(labels)) * 100
        bar = "█" * int(percentage / 5)
        print(f"  {class_name:12} : {count:3} ({percentage:5.1f}%) {bar}")
    
    # Estadísticas de longitud
    text_lengths = [len(text.split()) for text in texts]
    
    print(f"\n📝 Estadísticas de Longitud (palabras):")
    print(f"  Mínimo:    {min(text_lengths)} palabras")
    print(f"  Máximo:    {max(text_lengths)} palabras")
    print(f"  Promedio:  {sum(text_lengths)/len(text_lengths):.1f} palabras")
    print(f"  Mediana:   {sorted(text_lengths)[len(text_lengths)//2]} palabras")
    
    # Longitud por clase
    print(f"\n📝 Longitud Promedio por Clase:")
    for class_name in sorted(set(labels)):
        class_texts = [text for text, label in zip(texts, labels) if label == class_name]
        class_lengths = [len(text.split()) for text in class_texts]
        avg_length = sum(class_lengths) / len(class_lengths)
        print(f"  {class_name:12} : {avg_length:6.1f} palabras")
    
    # Balance del dataset
    print(f"\n⚖️  Balance del Dataset:")
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"  Ratio desbalance: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 2:
        print(f"  ⚠️  Dataset desbalanceado. Considere técnicas de balanceo.")
    elif imbalance_ratio > 1.5:
        print(f"  ⚠️  Ligero desbalance detectado.")
    else:
        print(f"  ✓ Dataset bien balanceado.")
    
    # Ejemplos
    print(f"\n📋 Ejemplos de Reseñas por Clase:")
    for class_name in sorted(set(labels)):
        example = next(text for text, label in zip(texts, labels) if label == class_name)
        preview = (example[:60] + "...") if len(example) > 60 else example
        print(f"  {class_name:12} : \"{preview}\"")
    
    print("="*70 + "\n")


def analyze_model_performance(results_dict):
    """Analiza performance del modelo."""
    
    print("\n" + "="*70)
    print("ANÁLISIS DE PERFORMANCE DEL MODELO")
    print("="*70)
    
    print(f"\n✓ Modelo: {results_dict['model_type'].upper()}")
    print(f"\n📊 Métricas Principales:")
    print(f"  Accuracy:  {results_dict['accuracy']:.4f} ({results_dict['accuracy']*100:.2f}%)")
    print(f"  Precision: {results_dict['precision']:.4f}")
    print(f"  Recall:    {results_dict['recall']:.4f}")
    print(f"  F1-Score:  {results_dict['f1']:.4f}")
    
    print(f"\n📈 Validación Cruzada (5-fold):")
    print(f"  F1-Score Medio: {results_dict['cv_mean']:.4f}")
    print(f"  Desv. Estándar: {results_dict['cv_std']:.4f}")
    print(f"  Scores por fold: {', '.join([f'{s:.4f}' for s in results_dict['cv_scores']])}")
    
    if results_dict['cv_std'] < 0.05:
        print(f"  ✓ Modelo estable (std < 0.05)")
    else:
        print(f"  ⚠️  Alto variance entre folds (std >= 0.05)")
    
    # Matriz de confusión
    cm = results_dict['confusion_matrix']
    print(f"\n🎯 Matriz de Confusión:")
    print(f"  {cm}")
    
    print("="*70 + "\n")


def compare_embedding_quality():
    """Analiza calidad de los embeddings."""
    
    print("\n" + "="*70)
    print("ANÁLISIS DE EMBEDDINGS")
    print("="*70)
    
    texts, labels = load_csv_data("data/raw/reviews.csv")
    
    embedder = Embedder()
    embeddings = embedder.encode(texts[:5])  # Probar con primeros 5
    
    print(f"\n🔢 Características de Embeddings:")
    print(f"  Modelo: SentenceTransformer (all-MiniLM-L6-v2)")
    print(f"  Dimensionalidad: {embeddings.shape[1]}")
    print(f"  Rango de valores: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    print(f"  Media: {embeddings.mean():.4f}")
    print(f"  Desviación estándar: {embeddings.std():.4f}")
    
    # Similaridad entre ejemplos de la misma clase
    print(f"\n📍 Validar separabilidad por clase:")
    print(f"  ✓ Con buenos embeddings, textos similares tendrán embeddings similares")
    print(f"  ✓ Textos de clases diferentes tendrán embeddings diferentes")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Ejecutar análisis
    analyze_dataset()
    
    # Análisis de embeddings
    if os.path.exists("data/raw/reviews.csv"):
        compare_embedding_quality()
    
    print("""
💡 INTERPRETACIÓN DE RESULTADOS:

✓ DATASET SALUDABLE (< 1.5x desbalance):
  - Modelos entrenarán sin problemas
  - Métrica accuracy es confiable
  
⚠️ DATASET DESBALANCEADO (> 1.5x):
  - Usar F1-Score en lugar de accuracy
  - Considerar class_weight='balanced' en modelos
  - Técnicas: SMOTE, oversampling, o ajustar threshold

✓ HIGH CV STABILITY (std < 0.05):
  - Modelo será consistente en producción
  - Confiable para despliegue
  
⚠️ HIGH VARIANCE (std >= 0.05):
  - Resultados pueden variar según datos
  - Más datos de entrenamiento recomendado
  - Riesgo para despliegue en producción

✓ HIGH F1-SCORE (> 0.80):
  - Modelo listo para producción
  - Depuración limitada necesaria
  
⚠️ MODERATE F1 (0.65-0.80):
  - Aceptable para MVP
  - Monitore en producción
  
❌ LOW F1 (< 0.65):
  - Mejorar modelo antes de despliegue
  - Expandir dataset
  - Ajustar hiperparámetros
"""
    )
