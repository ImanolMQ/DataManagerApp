# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:10:15 2024

@author: Imanol
"""

import streamlit as st
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
                    st.dataframe(df_nulls, key=key, height=self.height)
                else:
                    st.dataframe(filter_dataframe(self.dm.df.copy(), key=key),
                                 height=self.height)
        
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
        change_type = st.radio("Selecciona cómo cambiar el tipo de las columnas:", 
                               ('Múltiples columnas', 'Columna específica'))

        if change_type == 'Múltiples columnas':
            self._change_type_multiple_columns(col_options)
        else:
            self._change_type_single_column(col_options)
        st.table(self.dm.df.dtypes.to_frame().T)

    def _change_type_multiple_columns(self, col_options):
        # Cambio de tipo para múltiples columnas
        cols = st.multiselect("Selecciona columnas a cambiar de tipo:", col_options)
        new_type = st.selectbox("Selecciona el nuevo tipo:", 
                                ['category', 'int', 'float', 'bool', 'datetime', 'string'])
        date_format = st.text_input('Formato de fecha (Ej: %m/%d/%Y)') if new_type == 'datetime' else None
        
        if st.button("Cambiar tipos de las columnas", use_container_width=True) and cols:
            self.dm.change_type(cols, new_type, date_format=date_format)

    def _change_type_single_column(self, col_options):
        # Cambio de tipo para una sola columna con opciones personalizadas
        col = st.selectbox("Selecciona la columna:", col_options)
        new_type = st.selectbox("Selecciona el nuevo tipo:", 
                                ['category', 'int', 'float', 'bool', 'datetime', 'string'])
        new_categories = self._get_custom_categories(col) if new_type == 'category' else None
        date_format = st.text_input('Formato de fecha (Ej: %m/%d/%Y)') if new_type == 'datetime' else None
        
        if st.button("Cambiar tipo de la columna") and col:
            self.dm.change_type(col, new_type, new_categories=new_categories, date_format=date_format)

    def _get_custom_categories(self, col):
        # Método para capturar nombres personalizados de categorías
        categories = self.dm.df[col].unique().tolist()
        new_categories = [st.text_input(f"Nuevo nombre para '{val}':", val) or str(val) for val in categories]
        return new_categories if len(new_categories) == len(categories) else None

    def edit_null_values(self):
        
        # Reestructuración de imputación y eliminación de valores nulos
        options = {
            'Elimina filas': (self.dm.drop_na, 0),
            'Elimina columnas': (self.dm.drop_na, 1),
            'Imputar con la media': self._impute_vals(0),
            'Imputar con la mediana': self._impute_vals(1),
            'Imputar con la moda': self._impute_vals(2),
            'Imputar con valor anterior': self._impute_vals(3)
        }
        method = st.radio("Selecciona cómo tratar valores nulos:",
                          list(options.keys()))
        
        if method.split()[0] == 'Elimina':
            evaluation_methods = ['any (Hay valor nulo en alguna de las columnas seleccionadas)',
                                  'all (Todas las columnas seleccionadas son nulas)']
            selected_evaluation = st.selectbox("Especifica el método de evaluación:", 
                                         evaluation_methods)
            selected_evaluation = selected_evaluation.split()[0]
                
            cols = self.dm.tot_columns
            text = ("Especifica las columnas a las que" +
                    " tratar los valores nulos (si no se" +
                    " escogen se tratarán todas las columnas):")
            if selected_evaluation == 'any':
                if method == 'Elimina filas':
                    text = ('Especifica que columnas quieres ver si tienen algun ' +
                            'valor nulo en los registros:')
                if method == 'Elimina columnas':
                    text = ('Especifica que columnas quieres eliminar si tienen algun ' +
                            'valor nulo:')
            if selected_evaluation == 'all':
                if method == 'Elimina filas':
                    text = ('Especifica que columnas quieres ver que coincide que ' +
                            'tengan valores nulos a la vez en un registro:')
                if method == 'Elimina columnas':
                    text = ('Especifica que columnas quieres eliminar si tienen' +
                            ' todos los valores nulos:')
        if method.split()[0] == 'Imputar':
            text = ('Especifica en que columnas quieres imputar valores:')
            selected_evaluation = 'any'
            cols = self.dm.num_columns
            
        selected_cols = st.multiselect(text,
                                       cols)
        selected_cols = cols if not selected_cols else selected_cols
        
        # if st.checkbox('Mostrar filas con valores nulos de las columnas' +
        #                 ' introducidas'):
        #st.dataframe(self.dm.show_nulls(selected_cols, selected_evaluation))
        
        if st.button("Aplicar", use_container_width=True):
            func, axis = options[method]
            func(axis, selected_cols, selected_evaluation)
            
        return self.dm.show_nulls(selected_cols, selected_evaluation)

    def _impute_vals(self, imp_type):
        return lambda cols: self.dm.input_vals(cols, imp_type)
                    
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
        self.show_duplicates()
        if st.button("Eliminar duplicados", use_container_width=True):
            self.dm.drop_duplicates()

    def show_duplicates(self):
        df_dups = self.dm.show_duplicates()
        st.header(f"Registros duplicados: {len(df_dups)}")
        st.dataframe(df_dups, height=373)

    def normalize_and_standarize(self):
        options = {
            'Normalizar': self.normalize,
            'Estandarizar': self.standarize
        }
        selected_option = st.radio("Seleccione la opción:", list(options.keys()))
        options[selected_option]()

    def normalize(self):
        cols = st.multiselect("Selecciona columnas a normalizar:", self.dm.num_columns)
        if st.button('Normalizar'):
            self.dm.normalize_columns(cols)

    def standarize(self):
        cols = st.multiselect("Selecciona columnas a estandarizar:", self.dm.num_columns)
        if st.button('Estandarizar'):
            self.dm.standarize_columns(cols)

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