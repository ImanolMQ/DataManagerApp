# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:38:23 2024

@author: Imanol
"""

import pandas as pd
import streamlit as st
from src.auxiliar import ExtraInformation

# Configuración inicial
st.set_page_config(page_title="Carga de Dataset")
st.title("Carga del Dataset")

# Inicialización de estado de sesión
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Instancias y listas útiles
xtra_info = ExtraInformation()
encoding_options = [c['Codec'] for c in xtra_info.codecs_info]

# Parámetros de carga
uploaded_file = st.file_uploader("Selecciona un archivo", type=["csv",
                                                                "xlsx",
                                                                "json"])  # Soporte para múltiples formatos
separator = st.text_input('Separador (por defecto ";"):', value=";")
encoding = st.selectbox("Selecciona el encoding:", options=encoding_options,
                        index=encoding_options.index('utf_8'))
header_row = st.checkbox('Con cabecera', value=True)

# Opciones adicionales
if st.checkbox("Más opciones"):
    index_col = st.text_input("Columna para usar como índice (opcional):",
                              value="")
    if header_row:
        header_row = st.number_input("Fila para cabecera:", min_value=0, step=1,
                                     format="%i", value=0)
        
    rows_to_load = st.number_input("Número de filas a cargar (0 para todas):",
                                   min_value=0, value=0)

    # Opción para manejar valores nulos
    handle_nans = st.selectbox("Cómo manejar valores nulos:", ["Ninguno",
                                                               "Eliminar filas",
                                                               "Rellenar con valor"])
    fill_value = None
    if handle_nans == "Rellenar con valor":
        fill_value = st.text_input("Valor para rellenar:", value="")

else:
    index_col = None
    header_row = 0 if header_row else None
    rows_to_load = 0
    handle_nans = "Ninguno"
    fill_value =  None

header_row = 0 if header_row == False else header_row

# Botón para cargar el archivo
if st.button("Cargar DataFrame"):
    if uploaded_file:
        try:
            # Determinar el tipo de archivo y cargar el DataFrame
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, sep=separator or ";",
                                 encoding=encoding, index_col=index_col or None,
                                 header=header_row)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, index_col=index_col or None,
                                   header=header_row)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file, lines=True)

            # Cargar solo el número especificado de filas
            if rows_to_load > 0:
                df = df.head(rows_to_load)

            # Manejo de valores nulos
            if handle_nans == "Eliminar filas":
                df = df.dropna()
            elif handle_nans == "Rellenar con valor" and fill_value is not None:
                df = df.fillna(fill_value)

            # Almacenar el DataFrame en el estado de sesión
            st.session_state.data_loaded, st.session_state.df = True, df
            st.success("DataFrame cargado correctamente.")
            
            # Vista previa del DataFrame
            st.write("Vista previa del DataFrame:")
            st.dataframe(df.head(), use_container_width=True)

        except Exception as e:
            st.error(f"No se pudo cargar el DataFrame. Error: {e}")
    else:
        st.warning("Por favor, selecciona un archivo.")