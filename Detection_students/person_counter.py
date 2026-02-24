"""
Sistema de Detección y Conteo de Personas en Aulas para Registro de Asistencia
Utiliza YOLO v8 para detectar y contar personas con sus niveles de confianza
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO no está instalado. Instálalo con: pip install ultralytics")


class PersonCounterYOLO:
    """
    Clase para detectar y contar personas en imágenes usando YOLO v8
    """
    
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Inicializar el contador de personas
        
        Args:
            model_name: Nombre del modelo YOLO a usar (yolov8n, yolov8s, yolov8m, etc.)
            confidence_threshold: Umbral de confianza mínimo para aceptar detecciones
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO no está disponible. Instálalo con: pip install ultralytics")
        
        self.confidence_threshold = confidence_threshold
        logger.info(f"Cargando modelo YOLO: {model_name}")
        self.model = YOLO(model_name)
        self.person_class_id = 0  # En COCO dataset, persona = clase 0
        logger.info("Modelo YOLO cargado exitosamente")
    
    def detect_and_count_persons(self, 
                                 image_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Detectar y contar personas en una imagen
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Tupla con:
            - imagen anotada (np.ndarray)
            - diccionario con resultados de detección
        """
        # Leer la imagen
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
        else:
            # Si es un array numpy (desde Streamlit)
            image = image_path
        
        # Realizar detección
        logger.info("Ejecutando detección de YOLO...")
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        # Procesar resultados
        detections = results[0]
        boxes = detections.boxes
        
        # Filtrar solo personas
        person_detections = []
        for i, box in enumerate(boxes):
            cls = int(box.cls.item())
            if cls == self.person_class_id:  # Solo personas
                conf = float(box.conf.item())
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                person_detections.append({
                    'bbox': bbox,
                    'confidence': conf,
                    'class': cls,
                    'index': i
                })
        
        # Dibujar detecciones en la imagen
        image_annotated = image.copy()
        for detection in person_detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Dibujar caja delimitadora
            color = (0, 255, 0)  # Verde
            cv2.rectangle(image_annotated, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar etiqueta con confianza
            label = f"Persona {confidence:.2%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image_annotated, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            cv2.putText(image_annotated, label, 
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Preparar resultados
        results_dict = {
            'total_persons': len(person_detections),
            'detections': person_detections,
            'average_confidence': np.mean([d['confidence'] for d in person_detections]) if person_detections else 0.0,
            'min_confidence': min([d['confidence'] for d in person_detections]) if person_detections else 0.0,
            'max_confidence': max([d['confidence'] for d in person_detections]) if person_detections else 0.0,
            'image_shape': image.shape
        }
        
        return image_annotated, results_dict
    
    def print_results(self, results: Dict) -> None:
        """
        Imprimir resultados de detección de forma legible
        
        Args:
            results: Diccionario con resultados de detección
        """
        print("\n" + "="*70)
        print("REPORTE DE DETECCIÓN DE PERSONAS EN AULA".center(70))
        print("="*70)
        print(f"\nTotal de personas detectadas: {results['total_persons']}")
        print(f"Confianza promedio: {results['average_confidence']:.2%}")
        print(f"Confianza mínima: {results['min_confidence']:.2%}")
        print(f"Confianza máxima: {results['max_confidence']:.2%}")
        print(f"Dimensiones de imagen: {results['image_shape']}")
        
        if results['detections']:
            print("\nDetecciones individuales:")
            print("-" * 70)
            for i, detection in enumerate(results['detections'], 1):
                x1, y1, x2, y2 = detection['bbox']
                print(f"Persona {i}:")
                print(f"  - Confianza: {detection['confidence']:.2%}")
                print(f"  - Caja: ({x1}, {y1}) a ({x2}, {y2})")
        print("="*70 + "\n")
    
    def process_batch(self, image_folder: str) -> Dict:
        """
        Procesar múltiples imágenes en una carpeta
        
        Args:
            image_folder: Ruta a la carpeta con imágenes
            
        Returns:
            Diccionario con resultados de todas las imágenes
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [p for p in Path(image_folder).glob('*') 
                      if p.suffix.lower() in image_extensions]
        
        if not image_paths:
            logger.warning(f"No se encontraron imágenes en {image_folder}")
            return {}
        
        batch_results = {}
        for image_path in image_paths:
            logger.info(f"Procesando: {image_path.name}")
            try:
                _, results = self.detect_and_count_persons(str(image_path))
                batch_results[image_path.name] = results
            except Exception as e:
                logger.error(f"Error al procesar {image_path.name}: {e}")
        
        return batch_results


def main():
    """
    Ejemplo de uso del sistema de detección de personas
    """
    import sys
    
    # Crear instancia del contador
    counter = PersonCounterYOLO(
        model_name="yolov8n.pt",  # Nano model (más rápido)
        confidence_threshold=0.5
    )
    
    # Ejemplo: procesar una imagen
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Usar una imagen de ejemplo (debes proporcionar una imagen real)
        print("Uso: python person_counter.py <ruta_imagen>")
        print("Ejemplo: python person_counter.py classroom.jpg")
        return
    
    try:
        print(f"Procesando imagen: {image_path}")
        image_annotated, results = counter.detect_and_count_persons(image_path)
        
        # Mostrar resultados
        counter.print_results(results)
        
        # Guardar imagen anotada
        output_path = Path(image_path).stem + "_detected.jpg"
        cv2.imwrite(output_path, image_annotated)
        print(f"Imagen anotada guardada en: {output_path}")
        
        # Mostrar imagen (opcional)
        # cv2.imshow("Personas Detectadas", image_annotated)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")


if __name__ == "__main__":
    main()
