# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:11:32 2024

@author: Imanol
"""

import os
import pandas as pd
from src.data_manager import DataManager
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

def filter_dataframe(df, key, cut=False, show_selec_cols=True):
    
    col_options = DataManager(df).tot_columns
    
    if show_selec_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            modify = st.checkbox("Añadir filtros", key=key+'_checkbox')
        with col2:
    
            selected_cols = st.multiselect("Escoger columnas especificas a observar:",
                                           col_options, key=key+'visual_selected')
            if not selected_cols:
                selected_cols = col_options
    else:
        modify = st.checkbox("Añadir filtros", key=key+'_checkbox')
        selected_cols = col_options

    if not modify and cut:
        return df[selected_cols].head(10)
    
    if not modify and not cut:
        return df[selected_cols]
    

    df = df.copy()

    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filtrar dataframe en ",
                                           sorted(df.columns),
                                           key=key+'multiselect')
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Valores para {column}",
                    df[column].unique(),
                    default=list(df[column].unique(),),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Valores para {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Valores para {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring o regex en {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df[selected_cols]

def save_df(df):
    file = st.text_input("Escribe el nombre del archivo (sin .csv): ")
    path = st.text_input("Escribe la ruta de guardado (si no escribes nada" +
                         " se guardará en la carpeta donde este el archivo): ")
    if st.button("Guardar dataframe"):
        if file:
            file = file + '.csv'
            file = "./" + file if not path else os.path.join(path, file)
            file = file
            
            DataManager(df).save_df(file)
            st.success(f"Guardado en {file}.")
        else:
            st.error("Falta poner un nombre al archivo")

def change_files_name(directory, language):
    eng = ['1_Initial_Exploration.py','2_Data_Cleaning.py',
           '3_Data_Visualization.py','4_3D_Visualization.py','5_Save.py']
    sp = ['1_Exploración_Inicial.py','2_Limpieza_de_Datos.py',
          '3_Visualización_de_Datos.py','4_Visualización_3D.py','5_Guardar.py']
    
    if language == 'spanish':
        translate = {k:v for k, v in zip(eng,sp)}
        
    if language == 'english':
        translate = {k:v for k, v in zip(sp,eng)}
     
    # Recorre cada archivo en el directorio
    for filename in os.listdir(directory):
        # Verifica si el archivo tiene extensión .py
        if filename.endswith(".py"):
            if (filename in eng and language == 'spanish' or
                filename in sp and language == 'english'):
                new_name = translate[filename]
                # Renombra el archivo
                os.rename(os.path.join(directory, filename),
                          os.path.join(directory, new_name))