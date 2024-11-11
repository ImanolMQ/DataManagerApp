# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:10:15 2024

@author: Imanol
"""

import pandas as pd
import streamlit as st
import math
from src.streamlit.auxiliar import filter_dataframe

class DataCleaner():
    
    def __init__(self, dm):
        self.dm = dm
        self.height = 425 # Altura estandar para el data frame
        
    def render(self):
        # Definir las pestañas y sus acciones correspondientes
        actions = [
            ("Manejo de Valores Faltantes", self.edit_null_values),
            ("Detección y Eliminación de Duplicados", self.drop_duplicates),
            ("Corrección de Tipos de Datos", self.change_type_columns),
            ("Normalización y Estandarización", self.normalize_and_standarize),
            ("Manejo de Outliers", self.handling_outliers),
            ("Corrección de Errores en Datos", self.strings_correction),
            ("Renombrado de Columnas", self.change_name_columns)
        ]

        # Crear las pestañas y ejecutar la acción correspondiente en cada una
        tabs = st.tabs([name for name, _ in actions])
        for idx, (_, action) in enumerate(actions):
            self._create_tab(tabs[idx], action, key=f'tab{idx}')

        return self.dm.df.copy()
    
    def _create_tab(self, tab, action, key):
        # Método auxiliar para crear una pestaña con dos columnas y ejecutar la
        # acción
        with tab:
            col1, col2 = st.columns([1, 2])
            with col1:
                if key == "tab0":
                    df_nulls, method , columns = action()
                else:
                    action()
            with col2:
                if key == "tab0":
                    conector = 'y' if method == 'all' else 'o'
                    text = ''
                    for i, name in enumerate(columns):
                        if not i == 0:
                            text += ' ' + conector
                        text +=  ' ' + name
                    st.markdown(f"Hay {df_nulls.shape[0]} registros que tienen" +
                                " valores nulos en" + text + ".")
                    st.dataframe(df_nulls, key=key, height=self.height,
                                 use_container_width=True)
                else:
                    st.dataframe(filter_dataframe(self.dm.df.copy(), key=key),
                                 height=self.height, use_container_width=True)
        
    def change_name_columns(self):
        options = self.dm.tot_columns
        selected_option = st.selectbox("Selecciona la columna que quieras" +
                                      " cambiar de nombre: ", options)
        new_name = st.text_input("Escribe el nuevo nombre: ")
        
        change_name = st.button("Cambiar nombre de la columna",
                                use_container_width=True)
        
        if change_name and new_name:
                self.dm.change_name_col(selected_option, new_name)
                st.success("Nombre cambiado con exito")
        elif change_name and not new_name:
                st.error("Falta poner un nuevo nombre")
                
    def change_type_columns(self):
    # Modularizamos el tipo de cambio seleccionado
        col_options = self.dm.tot_columns
        self._change_type(col_options)

    def _change_type(self, col_options):
        # Función común para manejar el cambio de tipo para una o más columnas
        cols = st.multiselect("Selecciona columnas a cambiar de tipo:", col_options)
        cols = cols if cols else col_options
        #cols = cols if len(cols) > 1 else [cols]
        
        # Crear la tabla de tipos
        dtype_table_container = st.empty()
        self._display_dtype_table(cols, dtype_table_container)

        # Selección del nuevo tipo
        new_type = st.selectbox("Selecciona el nuevo tipo:", 
                                sorted(['category', 'int64', 'float64',
                                        'boolean', 'datetime', 'string']))

        # Si el tipo es 'datetime', pedir el formato de la fecha
        date_format = None
        if new_type == 'datetime':
            text_data = ("Formato fecha (por defecto %m/%d/%Y)\n\n" +
                         "Ejemplos:\n\n" +
                         "02/25/2024 → %m/%d/%Y\n\n" +
                         "15-01-2022 → %d-%m-%Y\n\n" +
                         "2022/01/15 14:30:45 → %Y/%m/%d %H:%M:%S")
            date_format = st.text_input(text_data)
            
        new_categories = None
        try:
            if new_type == 'category' and len(cols) == 1:
                if st.checkbox("Cambiar nombre de las categorías"):
                    df_nulls, _, _ = self.dm.show_nulls(cols)
                    if df_nulls.empty:
                        new_categories = self._get_custom_categories(cols[0])
                    else:
                        st.error("Antes de cambiar los nombres de las " +
                                 "categorías, tratar los valores nulos.")
        except Exception as e:
            st.error(f"{e}")

        # Botón para aplicar el cambio de tipo
        if st.button("Cambiar tipo de las columnas", use_container_width=True):
            try:
                self.dm.change_type(cols, new_type,
                                    new_categories=new_categories,
                                    date_format=date_format)
                # Actualizar la tabla de tipos
                self._display_dtype_table(cols, dtype_table_container)
                st.success(f"Cambio a {new_type} correctamente realizado")
            except Exception as e:
                st.error(f"Error al cambiar tipo de columna: {e}")

    def _display_dtype_table(self, cols, container):
        # Mostrar la tabla de tipos de datos
        dtype_table = pd.DataFrame(self.dm.df[cols]).dtypes.to_frame().T
        st.session_state.dtype_table = dtype_table  # Actualizamos la tabla en session_state

        # Contenedor para actualizar la tabla
        with container:
            st.table(st.session_state.dtype_table)
        
    def _get_custom_categories(self, col):
        # Método para capturar nombres personalizados de categorías
        categories = self.dm.df[col].unique().tolist()
        try:
            new_categories = [st.text_input(f"Nuevo nombre para '{str(val)}':", val) or str(val) for val in categories]
        except Exception as e:
            raise RuntimeError(f"Error al obtener categorías personalizadas: {e}")
        return new_categories if len(new_categories) == len(categories) else None

    def edit_null_values(self):
        
        # Reestructuración de imputación y eliminación de valores nulos
        options = {
            'Eliminar filas': (self.dm.drop_na, 0),
            'Eliminar columnas': (self.dm.drop_na, 1),
            'Imputar con la media': (self._impute_vals, 0),
            'Imputar con la mediana':( self._impute_vals, 1),
            'Imputar con la moda': (self._impute_vals, 2),
            'Imputar con valor anterior': (self._impute_vals, 3),
            'Imputar con valor posterior': (self._impute_vals, 4),
            'Imputar valor fijo': (self._impute_vals, 5),
            'Imputar fecha': (self._impute_vals, 6),
        }
        method = st.selectbox("Selecciona cómo tratar valores nulos:",
                              list(options.keys()))
        cols = self.dm.tot_columns
        fix_value = None
        
        if method.split()[0] == 'Eliminar':
            txt_rows = ("Opción 1. Hay un valor nulo en alguna de las columnas seleccionadas\n\n" +
                        "Opción 2. Todas las columnas seleccionadas son nulas")
            txt_cols = ("Opción 1. La columna tiene algún valor nulo \n\n" +
                        "Opción 2. La columna tiene todos los valores nulos")
            txt_methods = txt_rows if method == 'Elimina filas' else txt_cols
            evaluation_methods = {'Opción 1':'any', 'Opción 2':'all'}
            selected_evaluation = st.selectbox("Especifica el método de evaluación:\n\n" +
                                               txt_methods,
                                               evaluation_methods.keys())
            selected_evaluation = evaluation_methods[selected_evaluation]
            txt_default = '(Si no se escogen se evaluarán todas las columnas):'
            text = ("Especifica las columnas a las que" +
                    " tratar los valores nulos " + txt_default)
            if selected_evaluation == 'any':
                if method == 'Elimina filas':
                    text = ('Especifica que columnas quieres ver si tienen algun ' +
                            'valor nulo en los registros ' + txt_default)
                if method == 'Elimina columnas':
                    text = ('Especifica que columnas quieres eliminar si tienen algun ' +
                            'valor nulo ' + txt_default)
            if selected_evaluation == 'all':
                if method == 'Elimina filas':
                    text = ('Especifica que columnas quieres ver que coincide que ' +
                            'tengan valores nulos a la vez en un registro '  + txt_default)
                if method == 'Elimina columnas':
                    text = ('Especifica que columnas quieres eliminar si tienen' +
                            ' todos los valores nulos ' + txt_default)                 
            
        if method.split()[0] == 'Imputar':
            text = ('Especifica en que columnas quieres imputar valores:')
            selected_evaluation = 'any'
            
        selected_cols = st.multiselect(text, cols)
        selected_cols = cols if not selected_cols else selected_cols
        
        returned_nulls = self.dm.show_nulls(selected_cols, selected_evaluation)
        returned_df = (self.dm.df.copy(), 'any', selected_cols)
        
        is_drop = method.split()[0] == 'Eliminar'
        returned = returned_nulls if is_drop else returned_df
        
        is_fix = method in ['Imputar valor fijo', 'Imputar fecha']
        if is_fix:
            if method == 'Imputar valor fijo':
                fix_value = st.text_input("Valor fijo para imputar:")
            
            if method == 'Imputar fecha':
                fix_value = st.text_input("Fecha fija para imputar:")
        
        if st.button("Aplicar", use_container_width=True):
            
            try:
                func, param = options[method]
                
                if method.split()[0] == 'Eliminar':
                    func(param, selected_cols, selected_evaluation)
                    returned = self.dm.show_nulls(selected_cols,
                                                  selected_evaluation)
                if method.split()[0] == 'Imputar':   
                    func(param, cols = selected_cols, fix_value = fix_value)
                    returned = (self.dm.df.copy(), 'any', selected_cols)
            except Exception as e:
                st.error(f"{e}")
            
        return returned

    def _impute_vals(self, imp_type, cols, fix_value):
        cols = cols if isinstance(cols, list) else [cols]
        for col in cols:
            self.dm.imput_vals(imp_type, cols, fix_value)
                    
    def show_df_with_rows_range(self):
        df = self.dm.df.copy()
        columns = self.dm.tot_columns
        selected_cols = st.multiselect("Selecciona las columnas que quieras ver:",
                                       columns, key='df_with_rows')
        rows_range = st.slider("Selecciona el rango de filas a mostrar",
                               0, self.dm.n_rows, (0, 10))
        
        df_filtered = df.iloc[rows_range[0]:rows_range[1],:]
        
        if selected_cols:
            st.dataframe(df_filtered[selected_cols])
        else:
            st.dataframe(df_filtered)
            
    def drop_duplicates(self):
        cols = st.multiselect('Selecciona conjunto de columnas para ver si ' +
                              'estan duplicadas:\n\n (Si no se selecciona ' +
                              'ninguna se comprobará si hay filas duplicadas)',
                              self.dm.tot_columns)
        cols = cols if cols else self.dm.tot_columns
        
        self.show_duplicates(cols)
        if st.button("Eliminar duplicados", use_container_width=True):
            self.dm.drop_duplicates(cols)

    def show_duplicates(self, cols=None):
        df_dups = self.dm.show_duplicates(cols)
        st.header(f"Registros duplicados: {len(df_dups)}")
        st.dataframe(df_dups, height=290, use_container_width=True)

    def normalize_and_standarize(self):
        options = {
            'Min-Max Scaling': {'func': self.dm.min_max_scaling,
                                'text':'Escala los datos al rango [0,1]'},
            'Min-Max Scaling Personalizado': {'func': self.dm.min_max_scaling,
                                              'text': 'Escala los datos al ' +
                                              'rango especificado'},
            'Z-score': {'func': self.dm.z_score_standardization,
                                'text': ('Normaliza a media 0 y desviación' +
                                         ' estándar de 1')},
            'MaxAbs Scaling': {'func': self.dm.max_abs_scaling,
                                'text': ('Escala los datos por el valor ' +
                                         'absoluto máximo, preservando ceros ' +
                                         'y signos.')},
            'Robust Scaling': {'func': self.dm.robust_scaling,
                                'text': ('Estandariza usando la mediana y el ' +
                                         'rango intercuartílico, siendo menos ' +
                                         'sensible a los outliers.')}
        }
        selected_option = st.radio("Seleccione la opción:", list(options.keys()))
        
        st.info(options[selected_option]['text'])
        
        is_personalized = selected_option == 'Min-Max Scaling Personalizado'
        if is_personalized:
            col1, col2 = st.columns(2)
            with col1:
                min_s = st.number_input('Minimo: ', value=0)
            with col2:
                max_s = st.number_input('Máximo: ', value=1)
                
        # st.markdown(self.dm.scales)
        
        cols = st.multiselect("Selecciona columnas en las que aplicar\n\n" + 
                              "(En caso de no seleccionar se aplica sobre todas):",
                              self.dm.num_columns)
        
        cols = cols if cols else self.dm.num_columns
        
        if st.button('Realizar escalado', use_container_width=True):
            if is_personalized:
                st.session_state.scales = options[selected_option]['func'](cols,
                                                                           min_s,
                                                                           max_s)
            else:
                st.session_state.scales = options[selected_option]['func'](cols)
            
        if st.button('Deshacer ultimo escalado de las columnas seleccionadas',
                     use_container_width=True):
            try:
                st.session_state.scales = self.dm.inverse_scaling(cols)
            except Exception as e:
                st.error(f'{e}')
                

    def handling_outliers(self):
        cols = st.selectbox("Selecciona columna para outliers:", self.dm.num_columns)
        n_outliers, outliers = len(self.dm.return_column_outliers(cols)[0]), self.dm.return_column_outliers(cols)[0]
        st.write(f"Tiene {n_outliers} outliers")
        st.dataframe(outliers, height=265, width=500)

        if st.button("Eliminar filas con outliers", use_container_width=True):
            self.delete_outliers(cols)
        elif st.button("Capar outliers", use_container_width=True):
            self.winsorizing_outliers(cols)

    def delete_outliers(self, col):
        self.dm.delete_column_outliers(col)

    def winsorizing_outliers(self, col):
        self.dm.winsorizing_column_outliers(col)

    def strings_correction(self):
        cols = st.selectbox("Selecciona columna de texto a corregir:", self.dm.return_string_columns())
        if st.button('Eliminar espacios', use_container_width=True):
            self.dm.string_strip(cols)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Minúsculas', use_container_width=True):
                self.dm.string_lower(cols)
        with col2:
            if st.button('Mayúsculas', use_container_width=True):
                self.dm.string_upper(cols)

        old_str, new_str = st.text_input('Texto a corregir'), st.text_input('Nuevo texto')
        if st.button("Reemplazar texto", use_container_width=True):
            self.dm.string_replace(cols, old_str, new_str)

# import pandas as pd
# import numpy as np

# # Crear un DataFrame de ejemplo para pruebas de normalización y estandarización
# np.random.seed(42)  # Fijar la semilla para reproducibilidad

# # Crear el DataFrame
# df = pd.DataFrame({
#     'Edad': np.random.randint(18, 65, size=100),               # Edad entre 18 y 65
#     'Ingresos': np.random.normal(50000, 15000, 100),           # Ingresos con media 50000 y desviación estándar 15000
#     'Calificación': np.random.uniform(1, 5, 100),              # Calificación en un rango de 1 a 5
#     'Experiencia Laboral (años)': np.random.randint(0, 40, 100) # Años de experiencia laboral
# })

# # Añadir una columna con valores típicos y algunos outliers
# # Generar valores de deuda normales en el rango 0 a 50000
# deuda_normal = np.random.normal(20000, 5000, 95)

# # Crear algunos outliers
# deuda_outliers = np.array([200000, 300000, 150000, 250000, 500000])

# # Combinar los valores normales con los outliers
# deuda = np.concatenate([deuda_normal, deuda_outliers])

# # Añadir la nueva columna al DataFrame
# df['Deuda'] = deuda

# df.to_csv('data/norm_std.csv', index=False, sep=";")