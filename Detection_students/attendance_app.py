"""
Aplicación Streamlit para Detección y Conteo de Personas en Aulas
Sistema de Registro de Asistencia usando Visión Artificial (YOLO)
"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from person_counter import PersonCounterYOLO
import io

# Configuración de la página
st.set_page_config(
    page_title="Detector de Personas - Asistencia UIDE",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown("""
<div class="header">
    <h1>👥 Sistema de Detección de Personas para Registro de Asistencia</h1>
    <p>UIDE - Utilizando Visión Artificial (YOLO v8)</p>
</div>
""", unsafe_allow_html=True)

# Cargar modelo (con caché)
@st.cache_resource
def load_person_counter():
    """Carga el modelo YOLO una sola vez"""
    try:
        return PersonCounterYOLO(
            model_name="yolov8n.pt",  # Modelo Nano (más rápido)
            confidence_threshold=0.5
        )
    except Exception as e:
        st.error(f"Error al cargar el modelo YOLO: {e}")
        return None


# Barra lateral
st.sidebar.title("⚙️ Configuración")

confidence_threshold = st.sidebar.slider(
    "Umbral de Confianza",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Confianza mínima para considerar una detección válida"
)

model_size = st.sidebar.selectbox(
    "Tamaño del Modelo YOLO",
    options=["yolov8n.pt (Nano - Rápido)", "yolov8s.pt (Pequeño)", "yolov8m.pt (Medio)"],
    index=0,
    help="Mayor precisión = Mayor tiempo de procesamiento"
)

# Mapear selección a nombre del modelo
model_map = {
    "yolov8n.pt (Nano - Rápido)": "yolov8n.pt",
    "yolov8s.pt (Pequeño)": "yolov8s.pt",
    "yolov8m.pt (Medio)": "yolov8m.pt"
}
selected_model = model_map[model_size]

# Intentar cargar el modelo
person_counter = load_person_counter()

if person_counter is None:
    st.error("❌ No se pudo cargar el modelo YOLO. Por favor, instala ultralytics con: pip install ultralytics")
else:
    # Actualizar umbral de confianza
    person_counter.confidence_threshold = confidence_threshold
    
    # Pestañas principales
    tab1, tab2, tab3 = st.tabs(["📸 Procesar Imagen", "📊 Resultados Detallados", "ℹ️ Información"])
    
    with tab1:
        st.subheader("Elige una fuente de imagen")
        
        # Opciones de entrada
        input_method = st.radio(
            "Selecciona cómo cargar la imagen:",
            options=["Subir archivo", "Usar URL", "Tomar foto (Webcam)"],
            horizontal=True
        )
        
        image_input = None
        image_name = None
        
        if input_method == "Subir archivo":
            uploaded_file = st.file_uploader(
                "Carga una imagen del aula",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                help="Imagen que contenga personas en el aula"
            )
            if uploaded_file is not None:
                # Leer el contenido del archivo antes de abrirlo
                image_input = Image.open(io.BytesIO(uploaded_file.read()))
                image_name = uploaded_file.name
        
        elif input_method == "Usar URL":
            url_input = st.text_input("Ingresa la URL de la imagen")
            if url_input:
                try:
                    import requests
                    response = requests.get(url_input)
                    image_input = Image.open(io.BytesIO(response.content))
                    image_name = "imagen_url.jpg"
                except Exception as e:
                    st.error(f"Error al cargar imagen desde URL: {e}")
        
        elif input_method == "Tomar foto (Webcam)":
            st.info("💡 Nota: Esta función requiere permisos de cámara")
            camera_image = st.camera_input("Toma una foto del aula")
            if camera_image is not None:
                image_input = Image.open(camera_image)
                image_name = "captura_webcam.jpg"
        
        # Procesar imagen si está disponible
        if image_input is not None:
            # Mostrar imagen original
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Imagen Original")
                st.image(image_input, use_column_width=True)
            
            # Procesar detección
            try:
                with st.spinner("🔍 Detectando personas... Por favor espera..."):
                    # Convertir PIL Image a numpy array para el procesamiento
                    image_array = cv2.cvtColor(
                        np.array(image_input), 
                        cv2.COLOR_RGB2BGR
                    )
                    
                    # Realizar detección
                    image_annotated, results = person_counter.detect_and_count_persons(image_array)
                    
                    # Convertir de vuelta a RGB para mostrar
                    image_annotated_rgb = cv2.cvtColor(image_annotated, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("Personas Detectadas")
                    st.image(image_annotated_rgb, use_column_width=True)
                
                # Mostrar resultados principales
                st.markdown("---")
                st.subheader("📊 Resultados del Análisis")
                
                # Métricas principales
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric(
                        "Total de Personas",
                        f"{results['total_persons']}",
                        delta=None,
                        delta_color="green"
                    )
                
                with metric_cols[1]:
                    st.metric(
                        "Confianza Promedio",
                        f"{results['average_confidence']:.1%}",
                        help="Promedio de confianza de todas las detecciones"
                    )
                
                with metric_cols[2]:
                    st.metric(
                        "Confianza Mínima",
                        f"{results['min_confidence']:.1%}",
                        help="Detección con menor confianza"
                    )
                
                with metric_cols[3]:
                    st.metric(
                        "Confianza Máxima",
                        f"{results['max_confidence']:.1%}",
                        help="Detección con mayor confianza"
                    )
                
                # Mensaje de asistencia
                st.markdown("")
                if results['total_persons'] > 0:
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>✅ Registrado:</strong> Se han detectado <strong>{results['total_persons']} personas</strong> 
                        en el aula con una confianza promedio de <strong>{results['average_confidence']:.1%}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ Aviso:</strong> No se detectaron personas en la imagen. 
                        Verifica que la imagen sea clara o ajusta el umbral de confianza.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mostrar tabla de detecciones
                if results['detections']:
                    st.subheader("Detecciones Individuales")
                    
                    # Crear tabla de datos
                    detections_data = []
                    for i, detection in enumerate(results['detections'], 1):
                        x1, y1, x2, y2 = detection['bbox']
                        detections_data.append({
                            "ID": i,
                            "Confianza": f"{detection['confidence']:.2%}",
                            "Posición X": f"{x1}-{x2}",
                            "Posición Y": f"{y1}-{y2}",
                            "Área": f"{(x2-x1) * (y2-y1)} px²"
                        })
                    
                    st.dataframe(detections_data, use_container_width=True)
                
                # Descuento de imagen procesada
                col_download = st.columns([1, 1, 1])
                with col_download[1]:
                    # Convertir imagen anotada para descarga
                    image_annotated_pil = Image.fromarray(image_annotated_rgb)
                    buf = io.BytesIO()
                    image_annotated_pil.save(buf, format="JPEG")
                    buf.seek(0)
                    
                    st.download_button(
                        label="📥 Descargar imagen anotada",
                        data=buf,
                        file_name=f"personas_detectadas_{image_name}",
                        mime="image/jpeg"
                    )
                
            except Exception as e:
                st.error(f"❌ Error al procesar la imagen: {e}")
                st.info("Asegúrate de que la imagen sea válida y el formato sea soportado.")
    
    with tab2:
        st.subheader("Detalles Técnicos de Detecciones")
        
        st.info("""
        **Información que se captura por cada persona detectada:**
        - Confianza: Qué tan seguro está el modelo de que es una persona (0-100%)
        - Caja delimitadora (Bbox): Coordenadas de la región donde se detectó la persona
        - Clase: Tipo de objeto detectado (en este caso, siempre "Persona")
        """)
        
        # Ejemplos de interpretación de resultados
        st.subheader("📖 Interpretación de Resultados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Confianza Alta (>80%)**")
            st.write("✅ Detección muy confiable, probablemente correcto")
        
        with col2:
            st.write("**Confianza Baja (50-80%)**")
            st.write("⚠️ Puede ser correcto, revisar manualmente si es crítico")
        
        # Tabla de documentación
        st.subheader("📋 Referencia de Salidas")
        
        documentation = {
            "Campo": [
                "Total de Personas",
                "Confianza Promedio",
                "Confianza Mínima",
                "Confianza Máxima",
                "Caja Delimitadora (x1, y1, x2, y2)"
            ],
            "Descripción": [
                "Número de personas detectadas en la imagen",
                "Promedio de confianza de todas las personas detectadas",
                "Persona con menor confianza",
                "Persona con mayor confianza",
                "Coordenadas de píxeles: esquina superior izquierda a inferior derecha"
            ]
        }
        
        st.table(documentation)
    
    with tab3:
        st.subheader("ℹ️ Acerca del Sistema")
        
        st.markdown("""
        ### 🎯 Propósito
        Este sistema utiliza **YOLO v8 (You Only Look Once)**, un modelo de detección 
        de objetos en tiempo real, para detectar y contar personas en imágenes tomadas 
        por cámaras fijas en aulas. Está diseñado para automatizar el proceso de 
        registro de asistencia en la UIDE.
        
        ### 🔧 Tecnología
        - **Modelo**: YOLO v8 (Ultralytics)
        - **Dataset de Entrenamiento**: COCO (Common Objects in Context)
        - **Clases detectadas**: 80 objetos diferentes, incluyendo "Persona" (clase 0)
        - **Framework**: PyTorch
        
        ### 📊 Cómo Funciona
        1. **Captura**: Se carga una imagen del aula
        2. **Detección**: El modelo YOLO identifica todas las personas en la imagen
        3. **Confianza**: Proporciona un nivel de confianza para cada detección
        4. **Conteo**: Cuenta automáticamente el número total de personas
        5. **Registro**: Los resultados pueden usarse para registrar asistencia
        
        ### ⚙️ Configuración Disponible
        - **Umbral de Confianza**: Ajusta qué tan confiado debe ser el modelo (50-100%)
        - **Tamaño del Modelo**: Elige entre Nano (rápido), Pequeño, o Medio (más preciso)
        
        ### ⚠️ Limitaciones
        - Las personas parcialmente visibles pueden no detectarse
        - Confianza baja con oclusiones (personas parcialmente tapadas)
        - Requiere iluminación adecuada
        - No diferencia entre estudiantes y otros en el aula
        
        ### 📝 Requisitos del Sistema
        - Internet para descargar el modelo YOLO en primera ejecución
        - Al menos 2GB de RAM disponible
        - GPU recomendado para procesamiento más rápido
        
        ### 🚀 Mejoras Futuras
        - Integración con base de datos de estudiantes
        - Seguimiento de personas (tracking) entre fotogramas
        - Análisis de video en tiempo real
        - Reconocimiento facial para asistencia personalizada
        - Generación de reportes automáticos
        """)
        
        st.markdown("---")
        st.write("**Versión**: 1.0")
        st.write("**Última actualización**: 2024")
        st.write("**Universidad**: UIDE")
