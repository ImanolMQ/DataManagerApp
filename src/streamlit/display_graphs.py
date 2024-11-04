# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:10:49 2024

@author: Imanol
"""

import streamlit as st
from src.data_manager import DataManager
from src.streamlit.auxiliar import filter_dataframe
from pandas.api.types import is_numeric_dtype


class GraphsDisplayer():
    def __init__(self, dm):
        self.dm = dm
        
    def show_plots(self, cols, key):
        df = self.dm.df.copy()
     
        if st.checkbox("Ver los graficos", key=key+'_graph'):
            cols = cols or st.multiselect("Selecciona las columnas a mostrar" +
                                          " en la descripción y los graficos:",
                                          options=sorted(self.dm.tot_columns),
                                          key=key+'_multiselect')
            
            cols = sorted(self.dm.tot_columns) if not cols else cols
                
            traspuesta = st.checkbox("Invertir graficos",
                                     key=key+'_traspuesta_graph')
            
            plot_options = (['histogram', 'boxplot', 'violinplot', 'heatmap',
                             'corr', 'pairplot'] if is_numeric_dtype(df[cols[0]])
                            else ['donut', 'value_counts'])
                
            # Mostrar captions y estadísticas de los datos numéricos
            seleccion = st.selectbox("Selecciona una opción:", plot_options,
                                     key=key+'_show_plots')
            
            # Diccionario de gráficos y funciones correspondientes para reducir if-else
            plot_funcs = {
                'histogram': lambda: DataManager(df[cols]).show_histograms(traspuesta),
                'boxplot': lambda: DataManager(df[cols]).show_boxplots(traspuesta),
                'violinplot': lambda: DataManager(df[cols]).show_violinplots(traspuesta),
                'heatmap': lambda: DataManager(df[cols]).show_heatmap_st(),
                'corr': lambda: st.table(df[cols].corr()),
                'donut': lambda: DataManager(df[cols]).show_donuts(traspuesta),
                'value_counts': lambda: DataManager(df[cols]).show_value_counts(traspuesta),
                'pairplot': lambda: DataManager(df[cols]).show_pairplot_st()
            }

            fig = plot_funcs[seleccion]()
            if fig:
                st.plotly_chart(fig, key=f"{key}_{seleccion}")

    def facet_grids(self, rows, category, hue, type_p, key='facet',
                    variable2=None, traspuesta=False):

        return self.dm.show_facet_grid_st(
            rows=rows, category=category, hue=hue, type_p=type_p,
            variable2=variable2, traspuesta=traspuesta
        )

    def show_data_exploration_graphs(self):
        col1, col2, col3, col4 = st.columns(4)
        
        plot_types_1 = ['histogram']#, 'kdeplot']
        plot_types_2 = ['scatterplot']
        plot_types = plot_types_1 + plot_types_2
        
        df = filter_dataframe(self.dm.df.copy(), key='exploration_graphs',
                              show_selec_cols=False)
        traspuesta = st.checkbox("Invertir graficos", key='traspuesta_exploration')
        
        n_cols = sorted(DataManager(df).num_columns)
        c_cols = sorted(DataManager(df).cat_columns)
        c_cols.insert(0, 'None')
        
        with col1:
            selection1 = st.multiselect("Variables numéricas principales:", n_cols)
            n_cols_2 = [item for item in n_cols if item not in selection1]
            
            if not selection1:
                selection1 = n_cols
                n_cols_2 = 'None'
     
        with col2:
            c_cols_1 = c_cols
            selection2 = st.selectbox("Variables categóricas por las que dividir:", c_cols_1)
            if selection2 == 'None':
                c_cols_2 = c_cols
            else:
                c_cols_2 = [item for item in c_cols if item not in selection2]
                
        with col3:
            selection3 = st.selectbox("Variables categóricas con las que colorear:",
                                      c_cols_2)

        with col4:
            if len(n_cols) == len(selection1):
                plot_types = plot_types_1
            selection4 = st.selectbox("Selecciona tipo de gráfica:", plot_types)
            
            if selection4 in plot_types_2:
                second_n = st.selectbox("Variable numérica secundaria:", n_cols_2)
            else:
                second_n = None         
            
        category = None if selection2 == 'None' else selection2
        hue = None if selection3 == 'None' else selection3
            
        if selection1:
            figs = self.facet_grids(
                rows=selection1, 
                category=category,
                hue=hue,
                type_p=selection4,
                variable2=second_n,
                traspuesta=traspuesta
            )
            
            if len(figs) < 2:
                st.plotly_chart(figs[0])
                return
            for i, fig in enumerate(figs):
                st.plotly_chart(fig)
            
    def show_data_exploration_3D(self):
        col1, col2, col3, col4 = st.columns(4)
        
        df = filter_dataframe(self.dm.df.copy(), key='exploration_3D',
                              show_selec_cols=False)
        t_cols = sorted(DataManager(df).tot_columns)
     
        t_cols.insert(0, 'None')
        
        with col1:
            selection1 = st.selectbox("Eje 1:", t_cols)
            selection2 = 'None'
            selection3 = 'None'
            selection4 = 'None'
        if not selection1 == 'None':
            with col2:
                t_cols_2 = [item for item in t_cols if item != selection1]
                selection2 = st.selectbox("Eje 2", t_cols_2)
                selection3 = 'None'
                selection4 = 'None'
            if not selection2 == 'None':
                with col3:
                    t_cols_3 = [item for item in t_cols if item not in [selection1,
                                                                        selection2]]
                    selection3 = st.selectbox("Eje 3:", t_cols_3)
                    selection4 = 'None'
                if not selection3 == 'None':
                    with col4:
                        t_cols_4 = [item for item in t_cols if item not in [selection1,
                                                                            selection2,
                                                                            selection3]]
                        selection4 = st.selectbox("Color:", t_cols_4)
        
            
        axis_1 = None if selection1 == 'None' else selection1
        axis_2 = None if selection2 == 'None' else selection2
        axis_3 = None if selection3 == 'None' else selection3
        axis_4 = None if selection4 == 'None' else selection4
        
        if axis_1 and axis_2 and axis_3:
            st.plotly_chart(DataManager(df).rotable_3d(axis_1, axis_2,
                                                             axis_3, axis_4))