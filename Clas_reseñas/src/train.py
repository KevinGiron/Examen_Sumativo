from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import joblib
import numpy as np

def train_model(X, y, model_type="logistic", test_size=0.2, cv_folds=5, verbose=True):
    """
    Entrena un modelo de clasificación con validación cruzada.
    
    Args:
        X: Matriz de embeddings
        y: Vector de etiquetas
        model_type: Tipo de modelo ('logistic', 'svm', 'random_forest')
        test_size: Proporción de datos de prueba
        cv_folds: Número de folds para validación cruzada
        verbose: Mostrar resultados detallados
        
    Returns:
        clf: Modelo entrenado
        results: Diccionario con métricas de evaluación
    """
    
    # Seleccionar modelo
    if model_type == "logistic":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "svm":
        clf = LinearSVC(max_iter=2000, random_state=42)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Modelo {model_type} no soportado")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Entrenar modelo
    if verbose:
        print(f"\n{'='*60}")
        print(f"Entrenando modelo: {model_type.upper()}")
        print(f"{'='*60}")
        print(f"Datos de entrenamiento: {len(X_train)}")
        print(f"Datos de prueba: {len(X_test)}")
    
    clf.fit(X_train, y_train)
    
    # Evaluación en conjunto de prueba
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    
    # Validación cruzada
    cv_scores = cross_val_score(clf, X, y, cv=cv_folds, scoring='f1_weighted')
    
    results = {
        'model_type': model_type,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_test': y_test,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0)
    }
    
    if verbose:
        print(f"\nMétricas en conjunto de prueba:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"\nValidación cruzada ({cv_folds} folds):")
        print(f"  F1-Score medio: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"\nReporte de clasificación:")
        print(results['classification_report'])
    
    return clf, results


def save_model(model, filepath):
    """Guarda el modelo entrenado."""
    joblib.dump(model, filepath)
    print(f"Modelo guardado en: {filepath}")


def load_trained_model(filepath):
    """Carga un modelo previamente entrenado."""
    return joblib.load(filepath)