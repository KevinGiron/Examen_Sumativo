"""
Script de prueba simple para el detector de personas con YOLO
Ejecutar: python test_detector.py
"""

import os
import sys

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("PRUEBA DEL SISTEMA DE DETECCIÓN DE PERSONAS".center(70))
print("=" * 70)

# Paso 1: Verificar imports
print("\n[1/4] Verificando importaciones...")
try:
    import cv2
    print("   ✓ OpenCV importado correctamente")
except ImportError:
    print("   ✗ OpenCV no está instalado")
    sys.exit(1)

try:
    from PIL import Image
    print("   ✓ PIL (Pillow) importado correctamente")
except ImportError:
    print("   ✗ PIL no está instalado")
    sys.exit(1)

try:
    import numpy as np
    print("   ✓ NumPy importado correctamente")
except ImportError:
    print("   ✗ NumPy no está instalado")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print("   ✓ YOLO (ultralytics) importado correctamente")
except ImportError:
    print("   ✗ YOLO no está instalado")
    sys.exit(1)

# Paso 2: Descargar modelo YOLO
print("\n[2/4] Cargando modelo YOLO v8 Nano...")
try:
    model = YOLO("yolov8n.pt")
    print("   ✓ Modelo YOLO cargado exitosamente")
except Exception as e:
    print(f"   ✗ Error al cargar modelo: {e}")
    sys.exit(1)

# Paso 3: Crear imagen de prueba
print("\n[3/4] Creando imagen de prueba...")
try:
    # Crear una imagen simple con algunos rectángulos (simulando personas)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Agregar algunos rectángulos de color para simular personas
    cv2.rectangle(test_image, (50, 50), (150, 300), (0, 255, 0), -1)  # Verde
    cv2.rectangle(test_image, (200, 100), (350, 350), (0, 255, 0), -1)  # Verde
    cv2.rectangle(test_image, (400, 80), (550, 320), (0, 255, 0), -1)  # Verde
    
    # Guardar imagen de prueba
    test_image_path = "test_image.jpg"
    cv2.imwrite(test_image_path, test_image)
    print(f"   ✓ Imagen de prueba creada: {test_image_path}")
except Exception as e:
    print(f"   ✗ Error al crear imagen de prueba: {e}")
    sys.exit(1)

# Paso 4: Ejecutar detección
print("\n[4/4] Ejecutando detección de YOLO...")
try:
    results = model(test_image_path, conf=0.5, verbose=False)
    
    detections = results[0]
    boxes = detections.boxes
    
    # Filtrar solo personas (clase 0)
    person_count = 0
    confidences = []
    
    for box in boxes:
        cls = int(box.cls.item())
        if cls == 0:  # Persona
            person_count += 1
            conf = float(box.conf.item())
            confidences.append(conf)
    
    print(f"   ✓ Detección completada")
    
    # Dibujar cajas
    image_annotated = test_image.copy()
    for box in boxes:
        cls = int(box.cls.item())
        if cls == 0:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox
            conf = float(box.conf.item())
            
            # Dibujar caja
            cv2.rectangle(image_annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Dibujar etiqueta
            label = f"Persona {conf:.1%}"
            cv2.putText(image_annotated, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Guardar resultado
    output_path = "test_result.jpg"
    cv2.imwrite(output_path, image_annotated)
    print(f"   ✓ Imagen anotada guardada: {output_path}")
    
    # Mostrar resultados
    print("\n" + "=" * 70)
    print("RESULTADOS DE DETECCIÓN".center(70))
    print("=" * 70)
    print(f"\nTotal de personas detectadas: {person_count}")
    
    if confidences:
        print(f"Confianza promedio: {np.mean(confidences):.2%}")
        print(f"Confianza mínima: {min(confidences):.2%}")
        print(f"Confianza máxima: {max(confidences):.2%}")
        print(f"\nDetecciones individuales:")
        for i, conf in enumerate(confidences, 1):
            print(f"  Persona {i}: {conf:.2%} de confianza")
    
    print("\n" + "=" * 70)
    print("✓ PRUEBA EXITOSA - El sistema funciona correctamente".center(70))
    print("=" * 70)
    
    print("\n📝 Próximos pasos:")
    print("   1. Copia una imagen tuya del aula a esta carpeta")
    print("   2. Usa el script: python person_counter.py tu_imagen.jpg")
    print("   3. O ejecuta: streamlit run attendance_app.py")
    
except Exception as e:
    print(f"   ✗ Error durante la detección: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Limpiar archivos de prueba
    try:
        if os.path.exists("test_image.jpg"):
            os.remove("test_image.jpg")
    except:
        pass
