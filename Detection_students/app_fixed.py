"""
Aplicación Streamlit para Detección de Personas - Versión Simplificada
Ejecutar: streamlit run app_fixed.py
"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import sys
import os

# Agregar directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from person_counter import PersonCounterYOLO

# Configuración de la página
st.set_page_config(
    page_title="Detector de Personas - UIDE",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Detector de Personas en Aulas")
st.markdown("Sistema de Asistencia usando Visión Artificial (YOLO v8)")

# Cargar modelo
@st.cache_resource
def load_model():
    try:
        return PersonCounterYOLO(model_name="yolov8n.pt", confidence_threshold=0.5)
    except Exception as e:
        return None

# Configuración en sidebar
st.sidebar.title("Configuración")
confidence = st.sidebar.slider("Umbral de Confianza", 0.0, 1.0, 0.5, 0.05)

# Cargar modelo
model = load_model()

if model is None:
    st.error("❌ Error al cargar el modelo YOLO")
else:
    model.confidence_threshold = confidence
    
    # Interfaz principal
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Carga tu imagen")
        uploaded = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded:
        try:
            # Leer imagen
            image_bytes = uploaded.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            with col1:
                st.image(image, caption="Imagen original", use_column_width=True)
            
            # Procesar
            with st.spinner("🔍 Procesando..."):
                image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                result_img, results = model.detect_and_count_persons(image_array)
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            # Mostrar resultados
            with col2:
                st.image(result_img_rgb, caption="Personas detectadas", use_column_width=True)
            
            st.markdown("---")
            
            # Métricas
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("👥 Total", results['total_persons'])
            m2.metric("📊 Promedio", f"{results['average_confidence']:.1%}")
            m3.metric("📉 Mínimo", f"{results['min_confidence']:.1%}")
            m4.metric("📈 Máximo", f"{results['max_confidence']:.1%}")
            
            # Mensaje
            if results['total_persons'] > 0:
                st.success(f"✅ Se detectaron {results['total_persons']} personas")
            else:
                st.warning("⚠️ No se detectaron personas")
            
            # Tabla
            if results['detections']:
                st.subheader("Detecciones")
                data = []
                for i, det in enumerate(results['detections'], 1):
                    x1, y1, x2, y2 = det['bbox']
                    data.append({
                        "ID": i,
                        "Confianza": f"{det['confidence']:.2%}",
                        "Ancho": x2-x1,
                        "Alto": y2-y1
                    })
                st.dataframe(data, use_container_width=True)
            
            # Descargar
            result_pil = Image.fromarray(result_img_rgb)
            buf = io.BytesIO()
            result_pil.save(buf, format="JPEG")
            buf.seek(0)
            st.download_button("📥 Descargar resultado", buf, "resultado.jpg", "image/jpeg")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
