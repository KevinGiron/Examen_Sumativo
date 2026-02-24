import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_fscore_support
)
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Modelo"):
    """
    Evalúa el modelo de clasificación con múltiples métricas.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        y_pred_proba: Probabilidades predichas (opcional)
        model_name: Nombre del modelo para reportes
        
    Returns:
        metrics: Diccionario con todas las métricas
    """
    
    # Convertir a numpy arrays para asegurar compatibilidad
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, zero_division=0),
    }
    
    # Métricas por clase
    unique_labels = np.unique(y_true)
    metrics['per_class_metrics'] = {}
    
    for label in unique_labels:
        mask = np.asarray(y_true == label)
        support = int(np.sum(mask))
        if support > 0:
            p, r, f, s = precision_recall_fscore_support(
                y_true, y_pred, labels=[label], zero_division=0
            )
            metrics['per_class_metrics'][label] = {
                'precision': float(p[0]),
                'recall': float(r[0]),
                'f1': float(f[0]),
                'support': int(s[0])
            }
    
    return metrics


def print_evaluation_report(metrics, model_name="Modelo"):
    """Imprime un reporte detallado de evaluación."""
    
    print(f"\n{'='*70}")
    print(f"REPORTE DE EVALUACIÓN: {model_name}")
    print(f"{'='*70}")
    
    print(f"\nMétricas Globales:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    print(f"\nMétricas por Clase:")
    for label, class_metrics in metrics['per_class_metrics'].items():
        print(f"  {label}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1:        {class_metrics['f1']:.4f}")
        print(f"    Support:   {class_metrics['support']}")
    
    print(f"\nMatriz de Confusión:")
    print(metrics['confusion_matrix'])
    
    print(f"\n{'='*70}")


def plot_confusion_matrix(cm, labels, title="Matriz de Confusión"):
    """Visualiza la matriz de confusión."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    return plt


def compare_models(results_list):
    """
    Compara múltiples modelos.
    
    Args:
        results_list: Lista de tuplas (nombre_modelo, resultados_diccionario)
    """
    
    print(f"\n{'='*70}")
    print(f"COMPARACIÓN DE MODELOS")
    print(f"{'='*70}\n")
    
    comparison_data = []
    
    for model_name, results in results_list:
        comparison_data.append({
            'Modelo': model_name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1']
        })
    
    # Mostrar comparación
    for data in comparison_data:
        print(f"{data['Modelo']}:")
        print(f"  Accuracy:  {data['Accuracy']:.4f}")
        print(f"  Precision: {data['Precision']:.4f}")
        print(f"  Recall:    {data['Recall']:.4f}")
        print(f"  F1-Score:  {data['F1-Score']:.4f}\n")
    
    # Determinar mejor modelo
    best_model = max(comparison_data, key=lambda x: x['F1-Score'])
    print(f"Mejor modelo según F1-Score: {best_model['Modelo']} ({best_model['F1-Score']:.4f})")


def assess_deployment_readiness(metrics, min_accuracy=0.85, min_f1=0.80):
    """
    Evalúa si el modelo está listo para despliegue.
    
    Args:
        metrics: Diccionario de métricas
        min_accuracy: Accuracy mínimo requerido
        min_f1: F1-Score mínimo requerido
        
    Returns:
        dict: Análisis de readiness
    """
    
    assessment = {
        'ready_for_deployment': True,
        'issues': [],
        'recommendations': []
    }
    
    if metrics['accuracy'] < min_accuracy:
        assessment['ready_for_deployment'] = False
        assessment['issues'].append(
            f"Accuracy ({metrics['accuracy']:.4f}) por debajo del mínimo ({min_accuracy})"
        )
    
    if metrics['f1'] < min_f1:
        assessment['ready_for_deployment'] = False
        assessment['issues'].append(
            f"F1-Score ({metrics['f1']:.4f}) por debajo del mínimo ({min_f1})"
        )
    
    # Análisis de balance entre clases
    class_distribution = {}
    for label, class_metrics in metrics['per_class_metrics'].items():
        class_distribution[label] = class_metrics['support']
    
    total_samples = sum(class_distribution.values())
    is_imbalanced = any(count / total_samples < 0.2 for count in class_distribution.values())
    
    if is_imbalanced:
        assessment['recommendations'].append(
            "Dataset desbalanceado detectado. Considere usar técnicas de balanceo (SMOTE, class_weight)."
        )
    
    # Chequear performance por clase
    worst_class = min(metrics['per_class_metrics'].items(), key=lambda x: x[1]['f1'])
    if worst_class[1]['f1'] < 0.70:
        assessment['recommendations'].append(
            f"Clase '{worst_class[0]}' tiene bajo rendimiento (F1: {worst_class[1]['f1']:.4f}). "
            "Considere recopilar más datos o ajustar el modelo."
        )
    
    return assessment


def print_deployment_analysis(assessment):
    """Imprime análisis de readiness para despliegue."""
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE READINESS PARA DESPLIEGUE")
    print(f"{'='*70}\n")
    
    if assessment['ready_for_deployment']:
        print("✓ MODELO LISTO PARA DESPLIEGUE")
    else:
        print("✗ MODELO NO LISTO PARA DESPLIEGUE")
        print("\nProblemas identificados:")
        for issue in assessment['issues']:
            print(f"  - {issue}")
    
    if assessment['recommendations']:
        print("\nRecomendaciones:")
        for rec in assessment['recommendations']:
            print(f"  - {rec}")
    
    print(f"\n{'='*70}")
