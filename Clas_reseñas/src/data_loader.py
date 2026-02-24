import os
import pandas as pd

def load_csv_data(csv_path):
    """
    Carga datos desde un archivo CSV con columnas 'text' y 'label'.
    
    Args:
        csv_path: Ruta al archivo CSV
        
    Returns:
        texts: Lista de textos
        labels: Lista de etiquetas
    """
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    return texts, labels

def load_imdb_data(data_dir):
    """
    Carga datos desde estructura de carpetas (pos/neg/neutral).
    
    Args:
        data_dir: Directorio raíz con carpetas de etiquetas
        
    Returns:
        texts: Lista de textos
        labels: Lista de etiquetas
    """
    texts = []
    labels = []

    for label in os.listdir(data_dir):
        path = os.path.join(data_dir, label)
        if os.path.isdir(path):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    with open(file_path, encoding="utf-8") as f:
                        texts.append(f.read())
                        labels.append(label)

    return texts, labels