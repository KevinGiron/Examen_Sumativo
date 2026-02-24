"""
Script para descargar una imagen de prueba desde internet
y probar el detector YOLO con ella
"""

import requests
import sys
import os
from pathlib import Path

def download_test_image():
    """Descarga una imagen de personas desde internet"""
    
    # URL de una imagen de personas de libre uso
    # Imagen: Grupo de personas en la calle
    url = "https://images.unsplash.com/photo-1552664730-d307ca884978?w=640&q=80"
    
    # Nombre del archivo a guardar
    output_file = "test_people.jpg"
    
    print(f"Descargando imagen de prueba desde: {url}")
    print(f"Guardando en: {output_file}")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Guardar imagen
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        file_size = os.path.getsize(output_file)
        print(f"\n✓ Imagen descargada exitosamente ({file_size} bytes)")
        print(f"Archivo guardado: {output_file}")
        
        return output_file
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Error al descargar: {e}")
        print("\nIntentando con otra URL...")
        
        # Intentar con otra fuente
        try:
            url2 = "https://images.pexels.com/photos/3765139/pexels-photo-3765139.jpeg"
            response = requests.get(url2, timeout=10)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Imagen descargada de fuente alternativa")
            return output_file
            
        except Exception as e2:
            print(f"✗ Error en fuente alternativa: {e2}")
            print("\nNo se pudo descargar. Intenta estos pasos manualmente:")
            print("1. Abre: https://www.pexels.com/search/people/")
            print("2. Descarga una imagen y guárdala como 'test_people.jpg' en esta carpeta")
            return None

if __name__ == "__main__":
    image_file = download_test_image()
    
    if image_file:
        print(f"\nAhora ejecuta:")
        print(f"python person_counter.py \"{image_file}\"")
