import os
import joblib
import numpy as np
from pathlib import Path

# Import local modules
from src.embeddings import Embedder

# Streamlit is imported only when the app runs (after install)

def load_model(model_path="models/sentiment_model_logistic.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Ejecuta primero: python main.py")
    model = joblib.load(model_path)
    return model


def load_embedder():
    return Embedder()


def predict_text(model, embedder, text):
    embedding = embedder.encode([text])
    pred = model.predict(embedding)[0]
    prob = None
    try:
        proba = model.predict_proba(embedding)[0]
        prob = float(np.max(proba))
    except Exception:
        prob = None
    return pred, prob


# UI only runs when executed as script (streamlit runs file as script)
if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(page_title="Clasificador de Reseñas", layout="centered")
    st.title("Clasificador de Reseñas — Positivo / Negativo / Neutral")

    st.markdown(
        """
        Escribe una reseña en el cuadro de texto y haz click en "Clasificar".

        - El modelo usa embeddings con `sentence-transformers` y un clasificador entrenado.
        - Si el modelo no está presente, ejecuta primero `python main.py`.
        """
    )

    example_text = "Producto excelente, muy satisfecho con la compra"
    text = st.text_area("Escribe la reseña aquí:", value=example_text, height=160)

    # Cargar modelo y embedder una vez y almacenarlos en session_state
    if 'model' not in st.session_state:
        try:
            with st.spinner('Cargando modelo y embeddings...'):
                st.session_state['model'] = load_model()
                st.session_state['embedder'] = load_embedder()
        except Exception as e:
            st.error(str(e))
            st.stop()

    if st.button("Clasificar"):
        model = st.session_state['model']
        embedder = st.session_state['embedder']
        with st.spinner('Generando embedding y prediciendo...'):
            pred, prob = predict_text(model, embedder, text)

        emoji = {'positivo': '😊', 'negativo': '😞', 'neutral': '😐'}.get(pred, '❓')
        if prob is not None:
            st.success(f"{emoji} {pred.upper()}  (confianza: {prob:.1%})")
        else:
            st.success(f"{emoji} {pred.upper()}  (confianza: N/A)")

        st.markdown("---")
        st.subheader("Detalles")
        st.write(f"Texto: {text}")
        if prob is not None:
            st.write(f"Confianza: {prob:.4f}")

    st.sidebar.markdown("**Ejecutar**")
    st.sidebar.write("1) Asegúrate de haber entrenado y guardado el modelo: `python main.py`")
    st.sidebar.write("2) Ejecuta: `streamlit run streamlit_app.py`")
    st.sidebar.write("3) Escribe reseñas y presiona 'Clasificar'")
