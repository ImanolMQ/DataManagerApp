# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:08:58 2024

@author: Imanol
"""
import streamlit as st
from modules.data_manager import DataManager
from modules.streamlit.auxiliar import filter_dataframe
from modules.streamlit.display_graphs import GraphsDisplayer
from pandas.api.types import is_numeric_dtype


class DataExplorer():
    
    def __init__(self, dm):
        self.dm = dm
    
    def render(self):
        tab1, tab2, tab3, tab4 = st.tabs(['Tabla de datos',
                                          'Descripción',
                                          'Datos numéricos',
                                          'Datos categóricos'])
                
        with tab1:
            st.dataframe(filter_dataframe(self.dm.df.copy(), key='tab1',
                                          cut=True))           
        with tab2:
            self.show_df_info()
            
        with tab3:
            df_num = self.dm.df_num.copy()
            self.show_columns(df_num)
            
        with tab4:
            df_cat = self.dm.df_cat.copy()
            self.show_columns(df_cat)
            
       
    def show_df_info(self):
        st.markdown(f'El DataFrame tiene {self.dm.n_rows} filas y ' +
                    f'{self.dm.n_columns} columnas.')
        st.markdown('<div style="text-align: center; font-size: 20px;"><u>' +
                    'Columnas numéricas</u></div>', unsafe_allow_html=True)
        st.markdown(f"{self.dm.num_columns}")
        st.markdown('<div style="text-align: center; font-size: 20px;"><u>' +
                    'Columnas categóricas</u></div>', unsafe_allow_html=True)
        st.markdown(f"{self.dm.cat_columns}")
    
    
    def show_description(self, df, key):
        options = sorted(df.columns.tolist())
        show_table = st.checkbox("Ocultar Tabla", key=key+'_hide_table')
        
        if show_table:
            return None
        
        traspuesta = st.checkbox("Invertir estadisticos",
                                 key=key+'_traspuesta_table')
        
        cols = st.multiselect("Selecciona las columnas a mostrar en la" +
                              " descripción y los graficos:",
                              options=options,
                              key=key+'_multiselect')
        
        cols = cols if cols else options
        
        description = DataManager(df[cols])
        
        if key == 'numeric':
            st.table(description.numerical_description(t_num=traspuesta))
            
        if key == 'category':
            st.table(description.categorical_description(t_cat=traspuesta))
    
        return cols
    
    def show_columns(self, df):
        if df.shape[1] == 0:
            st.markdown("NO HAY DATOS DE ESTE TIPO")
            return
        
        key = 'numeric' if is_numeric_dtype(df.iloc[:,0]) else 'category'
        cols = self.show_description(df, key)
        
        dg = GraphsDisplayer(self.dm)
        dg.show_plots(cols, key)