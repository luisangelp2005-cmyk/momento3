import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import os
from streamlit_option_menu import option_menu
from sklearn.metrics import confusion_matrix, accuracy_score

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Banking Analytics Pro", layout="wide")

# --- 2. CARGA DE MODELOS ---
@st.cache_resource
def cargar_recursos():
    try:
        modelo_log = joblib.load('modelo_logistico.pkl')
        modelo_ann = joblib.load('red_neuronal.pkl')
        scaler = joblib.load('escalador.pkl')
        columnas = joblib.load('columnas_modelo.pkl')
        return modelo_log, modelo_ann, scaler, columnas
    except FileNotFoundError:
        st.error("⚠️ No se encontraron los archivos .pkl. Por favor, corre primero 'entrenamiento_modelos.py'")
        return None, None, None, None

mod_log, mod_ann, sc, cols_entrenamiento = cargar_recursos()

# --- 3. CSS PARA DISEÑO PROFESIONAL ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    h1, h2, h3, p, label { color: #000000 !important; }
    div.stButton > button:first-child {
        background-color: #000000;
        color: #ffffff !important;
        border-radius: 4px;
        font-weight: bold;
        width: 100%;
    }
    div.stButton > button:first-child:hover { background-color: #ff4b4b; }
    .res-container {
        border-left: 5px solid #000000;
        background-color: #f8f9fa;
        padding: 25px;
        margin-top: 20px;
        border-radius: 0 10px 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. MENÚ DE NAVEGACIÓN ---
selected = option_menu(
    menu_title=None,
    options=["Predicción Individual", "Análisis por Lotes"],
    icons=["person-vcard", "database-fill-add"],
    orientation="horizontal",
    styles={
        "container": {"background-color": "#000000"},
        "nav-link": {"color": "#ffffff"},
        "nav-link-selected": {"background-color": "#ff4b4b"},
    }
)

# --- 5. LÓGICA DE SECCIONES ---

if selected == "Predicción Individual":
    st.header("Análisis")
    
    if mod_ann is None:
        st.warning("El modelo no está listo. Ejecuta el entrenamiento.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            edad = st.number_input("Edad", 18, 100, 30)
            trabajo = st.selectbox("Ocupación", ["admin.", "technician", "blue-collar", "management", "retired", "services"])
            balance = st.number_input("Saldo anual (Balance)", value=1500)
        with col2:
            vivienda = st.selectbox("¿Tiene Hipoteca?", ["yes", "no"])
            duracion = st.number_input("Duración de llamada(seg)", value=200)
   tipo_modelo = st.radio("Modelo", ["Regresión Logística", "Red Neuronal (MLP)"])

        if st.button("Generar Diagnóstico Real"):
            
            input_dict = {'age': edad, 'balance': balance, 'duration': duracion, 
                          'job': trabajo, 'housing': vivienda}
            df_input = pd.DataFrame([input_dict])
            df_input = pd.get_dummies(df_input).reindex(columns=cols_entrenamiento, fill_value=0)
            X_input = sc.transform(df_input)

            modelo_actual = mod_log if tipo_modelo == "Regresión Logística" else mod_ann
            pred = modelo_actual.predict(X_input)[0]
            prob = modelo_actual.predict_proba(X_input)[0][1] * 100

            st.markdown('<div class="res-container">', unsafe_allow_html=True)
            
            if prob >= 50:
                st.subheader("Resultado: INTERESADO")
                st.write(f"Confianza del Modelo: **{prob:.2f}%**")
                st.success("✅ Alta probabilidad de éxito comercial.")
            else:
                st.subheader("Resultado: NO INTERESADO")
                st.write(f"Confianza del Modelo: **{prob:.2f}%**")
                st.error("❌ Baja probabilidad de éxito comercial.")
                
            st.markdown('</div>', unsafe_allow_html=True)
          

elif selected == "Análisis por Lotes":
    st.header("Carga Masiva de Datos")
    archivo = st.file_uploader("Sube bank-full.csv", type=["csv"])
    
    if archivo:
        df = pd.read_csv(archivo, sep=None, engine='python')
        st.write(f"Registros cargados: {len(df)}")
        
        # --- MOSTRAR PRIMEROS 10 REGISTROS ---
        st.subheader("Vista Previa de Datos (Primeros 10)")
        st.dataframe(df.head(10))
        
        if st.button("Ejecutar Clasificación"):
            # Procesamiento de datos
            y_real = pd.Series(np.where(df['y'] == 'yes', 1, 0)) 
            X_masivo = df.drop('y', axis=1)
            X_masivo = pd.get_dummies(X_masivo).reindex(columns=cols_entrenamiento, fill_value=0)
            X_masivo_s = sc.transform(X_masivo)
            
            # --- PREDICCIÓN RED NEURONAL ---
            st.markdown("---")
            st.subheader("1. Evaluación Red Neuronal:  (ANN)")
            y_pred_ann = mod_ann.predict(X_masivo_s)
            
            col_a1, col_b1 = st.columns(2)
            with col_a1:
                cm_ann = confusion_matrix(y_real, y_pred_ann)
                fig_ann, ax_ann = plt.subplots()
                sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
                st.pyplot(fig_ann)
            
            with col_b1:
                acc_ann = accuracy_score(y_real, y_pred_ann)
                st.metric("Exactitud ANN", f"{acc_ann:.4f}")
                st.info("Resultados basados en la arquitectura de la Red Neuronal entrenada.")

            # --- PREDICCIÓN REGRESIÓN LOGÍSTICA ---
            st.markdown("---")
            st.subheader("2. Evaluación: Matrix De confusión - Regresión Logística")
            y_pred_log = mod_log.predict(X_masivo_s)
            
            col_a2, col_b2 = st.columns(2)
            with col_a2:
                cm_log = confusion_matrix(y_real, y_pred_log)
                fig_log, ax_log = plt.subplots()
                sns.heatmap(cm_log, annot=True, fmt='d', cmap='Greens', 
                            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
                st.pyplot(fig_log)
            
            with col_b2:
                acc_log = accuracy_score(y_real, y_pred_log)
                st.metric("Exactitud Logística", f"{acc_log:.4f}")
                st.info("Resultados basados en el modelo lineal de Regresión Logística.")
