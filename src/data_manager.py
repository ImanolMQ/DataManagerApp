# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:45:07 2024

@author: Imanol
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display
import plotly.graph_objs as go
from matplotlib.colors import to_hex
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import gaussian_kde

class DataManager():
    mi_paleta = {'Navy Blue': '#1b2e3c',
                 'Crimson': '#4b0000',
                 'Black': '#0c0c1e',
                 'Cream': '#f3e3e2'}
    data_types = []

    def __init__(self, df):
        self.df = self.process_df(df)
        self.tot_columns = self.df.columns.tolist()
        self.split_df()
        self.shape = self.df.shape
        self.n_rows = self.df.shape[0]
        self.n_columns = self.df.shape[1]
    
    @staticmethod
    def process_df(df):
        df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        df.columns = df.columns.str.strip()
        return df
      
    def split_df(self):
        self.df_num = self.df.select_dtypes(include="number")
        self.df_cat = self.df.select_dtypes(exclude="number")
        self.num_columns = self.df_num.columns.tolist()
        self.cat_columns = self.df_cat.columns.tolist()
        
    def show_info(self):
        self.df.info()
        
    def numerical_description(self, t_num=False):
        # Descripción de columnas numéricas
        des_num = self.df_num.describe().T
        des_num["Tipos"] = self.df_num.dtypes
        des_num["Nulos"] = self.df_num.isna().sum()
        des_num["Únicos"] = self.df_num.nunique()
        des_num = des_num.T if t_num else des_num

        return des_num
      
        
    def categorical_description(self, t_cat=False):
        # Descripción de columnas categóricas
        des_cat = self.df_cat.describe(include="all").T
        des_cat["Tipos"] = self.df_cat.dtypes
        des_cat["Nulos"] = self.df_cat.isna().sum()
        des_cat = des_cat.T if t_cat else des_cat
            
        return des_cat
            
    def show_describe(self, t_num=False, t_cat=False):
        """Mostrar descripción de los datos divididos en categoricos
        y numéricos
        """
        # Mostrar tamaño del DataFrame
        print("** Datos del DataFrame **")
        print(f"El DataFrame tiene {self.n_rows} filas y {self.n_columns} columnas.\n")

        # Mostrar las columnas
        print("** Columnas numéricas **")
        print(f"{self.num_columns}\n")
        
        print("** Columnas categóricas **")
        print(f"{self.cat_columns}\n")
        
        print("** Descripción de las columnas numéricas **")
        display(self.numerical_description(t_num))
        print("\n")
        
        print("** Descripción de las columnas categóricas **")
        display(self.categorical_description(t_cat))
        print("\n")

    def show_head(self, n=5):
        """Mostrar las primeras n filas del DataFrame"""
        print(f"** Primeras {n} filas del DataFrame **")
        return self.df.head(n)

    def show_tail(self, n=5):
        """Mostrar las ultimas n filas del DataFrame"""
        print(f"** Ultimas {n} filas del DataFrame **")
        return self.df.tail(n)

    def _configure_plot_simple(self, name, traspuesta, variable, axes, idx):

        mi_paleta = DataManager.mi_paleta

        plot_axe = 'y' if traspuesta else 'x'

        if name == 'histograma':
            plot_config = {plot_axe: self.df_num[variable],
                            'kde':True,
                            'color': mi_paleta['Navy Blue'],
                            'edgecolor': mi_paleta['Black'],
                            'ax':axes[idx]}
            sns.histplot(**plot_config)

        if name == 'boxplot':
            plot_config = {plot_axe: self.df_num[variable],
                           'color': mi_paleta['Black'],
                           'whiskerprops':{'color': mi_paleta['Cream'],
                                            'linewidth': 1},
                           'capprops':{'color': mi_paleta['Cream'], 'linewidth': 1},
                           'medianprops':{'color': mi_paleta['Cream'], 'linewidth': 1},
                           'flierprops':{
                               'marker': 'o',
                               'markerfacecolor': mi_paleta['Cream'],
                               'markersize': 6,
                               'markeredgecolor': mi_paleta['Black'],
                               'markeredgewidth': 0.5},
                           'ax':axes[idx]
                            }
            sns.boxplot(**plot_config)

        if name == 'violinplot':
            plot_config = {plot_axe: self.df_num[variable],
                           'color': mi_paleta['Navy Blue'],
                           'inner_kws':dict(box_width=15, whis_width=2,
                                 color=mi_paleta['Black']),
                           'ax':axes[idx]
                           }
            sns.violinplot(**plot_config)

        if name == 'value_counts':
            plot_config = {plot_axe:variable, 'data':self.df_cat,
                           'color': mi_paleta['Black'], 'ax':axes[idx]}
            sns.countplot(**plot_config)
            
        if name == 'donut':
            etiquetas = self.df_cat[variable].unique().tolist()
            conteos = self.df_cat[variable].value_counts()

            # Gráfico de pastel
            plot_config = {'x': conteos, 'labels': etiquetas,
                           'colors':sns.color_palette("bright"),
                           'textprops': {'color': mi_paleta['Black'],
                                         'fontsize':8}}
            axes[idx].pie(**plot_config)

            # Círculo central para el donut
            circulo_central = plt.Circle((0, 0), 0.7, color=mi_paleta['Crimson'])
            axes[idx].add_artist(circulo_central)

            # Leyenda
            #axes[idx].legend(labels=etiquetas, loc='upper left')
            
        if name == 'facetgrid':
            axes[idx] = sns.FacetGrid(self.df, col=self.category)
            axes[idx].map(sns.histplot, variable)
            
    def _configure_plot_simple_2(self, fig, type_name, traspuesta, variable,
                                 row, col, proportional):

        plot_axe = 'x'
        color='#FF4B4B'
        if type_name == 'histograma':
            plot_axe = 'y' if traspuesta else 'x'
            plot_config = {plot_axe:self.df_num[variable], 'name':variable,
                           'marker':dict(color=color)}
            
            fig.add_trace(
                go.Histogram(**plot_config),
                row=row, col=col
            )

        if type_name == 'boxplot':
            plot_axe = 'x' if traspuesta else 'y'
            plot_config = {plot_axe:self.df_num[variable], 'name':'',
                           'marker':dict(color='#FF4B4B')}
            
            fig.add_trace(
                go.Box(**plot_config),
                row=row, col=col
            )

        if type_name == 'violinplot':
            plot_axe = 'x' if traspuesta else 'y'
            plot_config = {plot_axe:self.df_num[variable], 'name':'',
                           'marker':dict(color='#FF4B4B'),
                           'line':dict(color='#FF4B4B'), 'box_visible':True,
                           'meanline_visible':True}
            fig.add_trace(
                go.Violin(**plot_config),
                row=row, col=col
            )

        if type_name == 'value_counts':
            counts = self.df_cat[variable].value_counts()
            n = self.df_cat[variable].shape[0]
            categories = counts.index.tolist()
            values =  counts.values * (1/n if proportional else 1)
            
            plot_axe_x = 'y' if traspuesta else 'x'
            plot_axe_y = 'x' if traspuesta else 'y'
            orientation = 'h' if traspuesta else 'v'
            plot_config = {plot_axe_x:categories,
                           plot_axe_y:values,
                           'orientation':orientation,
                           'marker':dict(color='#FF4B4B')}
            fig.add_trace(
                go.Bar(**plot_config),
                row=row, col=col
            )
            
            
        if type_name == 'donut':
            colors = ['#FF4B4B', '#31333F']
            
            labels = self.df[variable].value_counts().index.tolist()  # Etiquetas de categorías
            values = self.df[variable].value_counts().values  # Frecuencias de cada categoría
            ncolors = len(values)
            palette = DataManager.generate_gradient_colors(colors[0],
                                                                 colors[1],
                                                                 ncolors)
            min_percentage = 5
            total = sum(values)
            percentages = [(v / total) * 100 for v in values]
            
            text_template = [f"<b>{percent:.1f}%</b>"
                             if percent >= min_percentage else ""
                             for label, percent in zip(labels, percentages)]
            
            # Configuración para el diagrama de tarta
            plot_config = {
                'labels': labels,
                'values': values,
                'name': '',
                'textfont_size': 12,
                'textinfo': 'label+percent',
                'texttemplate': text_template,
                'marker': dict(colors=palette)
            }
            
            # Agregar el diagrama de tarta a la figura
            fig.add_trace(
                go.Pie(**plot_config),
                row=row, col=col
            )
            
            fig.update_traces(hole=.4, hoverinfo="label+percent+name")
            
        if type_name == 'facetgrid':
            pass
        
        if plot_axe == 'y' or (plot_axe == 'x' and type_name in ['boxplot','violinplot']):
            fig.update_xaxes(showgrid=True,# gridcolor='rgba(200, 200, 200, 0.5)',
                             row=row, col=col)
        else:
            fig.update_yaxes(showgrid=True,# gridcolor='rgba(200, 200, 200, 0.5)',
                             row=row, col=col)

    def _format_axes(self, axes, variable, idx, traspuesta, plot_type):
        mi_paleta = DataManager.mi_paleta

        # Ajustar fondo
        axes[idx].set_facecolor(mi_paleta['Crimson'])

        # Borrar ejes
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
        axes[idx].spines['left'].set_visible(False)
        axes[idx].spines['bottom'].set_visible(False)

        # Personalizar ticks
        axes[idx].tick_params(bottom=False, left=False)
        axes[idx].tick_params(axis='y', colors=mi_paleta['Cream'], labelsize=8)
        axes[idx].tick_params(axis='x', colors=mi_paleta['Cream'], labelsize=8)

        # Personalizar labels
        x, y = ('', variable) if traspuesta else (variable, '')
        axes[idx].set_ylabel(y, fontsize=10, color=mi_paleta['Cream'], fontstyle='italic')
        axes[idx].set_xlabel(x, fontsize=10, color=mi_paleta['Cream'], fontstyle='italic')

        # Cambiamos el valor de la traspuesta si las gaficas son violin o box
        traspuesta = not traspuesta if plot_type in ['violinplot', 'boxplot'] else traspuesta

        # Eliminar el grid o añadirlo
        axes[idx].set_axisbelow(True)
        if traspuesta:
            axes[idx].xaxis.grid(True, color=mi_paleta['Navy Blue'], linewidth=1)
            axes[idx].yaxis.grid(False)
        else:
            axes[idx].xaxis.grid(False)
            axes[idx].yaxis.grid(True, color=mi_paleta['Navy Blue'], linewidth=1)

    def _finalize_plot(self, suptitle):
        mi_paleta = DataManager.mi_paleta

        plt.suptitle(suptitle, fontsize=16, weight='bold', color=mi_paleta['Cream'], x=0.5, y=0.95)

        plt.subplots_adjust(top=0.85)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()

    def _show_plot(self, traspuesta, suptitle, type_name, cols):
        """Mostrar el tipo de gráfica solicitada"""
        categoricals = ['value_counts', 'donut']
 
        data = self.df_cat if type_name in categoricals else self.df_num
        n = len(data.columns)
        n_cols = min(4, n) if not cols else cols
        n_rows = int(np.ceil(n / n_cols))
        
        if n_cols == 1 and n_rows == 1:
            fig, axes = plt.subplots(2, 1, figsize=(15, 5 * n_rows))
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            
        axes = axes.ravel()

        # Fondo de la figura color personalizado
        fig.patch.set_facecolor(DataManager.mi_paleta['Crimson'])

        for idx, variable in enumerate(data.columns):
            self._configure_plot_simple(type_name, traspuesta, variable, axes, idx)
            self._format_axes(axes, variable, idx, traspuesta, type_name)

        for ax in axes[n:]:
            ax.set_visible(False)  # Ocultar ejes adicionales

        #self._finalize_plot(suptitle)
        return fig
    
    def _show_plot_2(self, traspuesta, suptitle, type_name, cols, proportional=False):
        """Mostrar el tipo de gráfica solicitada"""
        categoricals = ['value_counts', 'donut']
 
        data = self.df_cat if type_name in categoricals else self.df_num
        n = len(data.columns)
        n_cols = min(4, n) if not cols else cols
        n_rows = int(np.ceil(n / n_cols))
        
        specs = [[{'type': 'domain'} for _ in range(n_cols)] for _ in range(n_rows)]
        
        if type_name == 'donut':
            fig = make_subplots(rows=n_rows, cols=n_cols,
                                specs=specs,
                                subplot_titles=data.columns)
        else:
            fig = make_subplots(rows=n_rows, cols=n_cols,
                                subplot_titles=data.columns)

        for idx, variable in enumerate(data.columns):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            # Generar gráficos según el tipo de gráfico
            self._configure_plot_simple_2(fig, type_name, traspuesta,
                                          variable, row, col, proportional)

            
                
        fig.update_layout(
            showlegend=False,
            #plot_bgcolor=self.mi_paleta['Crimson'],  # Fondo personalizado
            #paper_bgcolor=self.mi_paleta['Crimson'],  # Fondo de la figura
            height=400 * n_rows,  # Ajustar la altura según el número de filas
            width=400 * n_cols   # Ajustar el ancho según el número de columnas
            )
        
        return fig

    def show_histograms(self, traspuesta=False, cols=None):
        return self._show_plot_2(traspuesta, "Frecuencia de Variables Numéricas",
                        "histograma", cols)

    def show_boxplots(self, traspuesta=True, cols=None):
        return self._show_plot_2(traspuesta, "Frecuencia de Variables Numéricas",
                        "boxplot", cols)

    def show_violinplots(self, traspuesta=True, cols=None):
        return self._show_plot_2(traspuesta, "Frecuencia de Variables Numéricas",
                        "violinplot", cols)

    def show_value_counts(self, traspuesta=False, cols=None, proportional=False):
        return self._show_plot_2(traspuesta, "Frecuencia de Variables Categóricas",
                        "value_counts", cols, proportional)

    def show_heatmap(self):
        """Mostrar mapa de calor de correlación"""
        mi_paleta = DataManager.mi_paleta

        # Crear una colormap personalizada con los colores deseados
        colores = [mi_paleta['Navy Blue'], mi_paleta['Black']]
        mi_cmap = LinearSegmentedColormap.from_list('mi_cmap', colores)

        corr = self.df_num.corr()

        fig, axes = plt.subplots()

        # Fondo de la figura color personalizado
        fig.patch.set_facecolor(DataManager.mi_paleta['Crimson'])

        heatmap = sns.heatmap(corr, cmap=mi_cmap, ax=axes)

        # Cambiar el color de los ticks
        axes.tick_params(left=False, bottom=False)
        plt.xticks(color=mi_paleta['Cream'], fontstyle='italic', fontsize=4)
        plt.yticks(color=mi_paleta['Cream'], fontstyle='italic', fontsize=4)

        # Acceder a la barra de color
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(colors=mi_paleta['Cream'], labelsize=4)

        #self._finalize_plot("Matriz de Correlación")
        
        return fig
    
    def show_heatmap_st(self):
        corr = self.df_num.corr()
        
        colorscale = [[0, "#31333F"],   # Azul para correlaciones negativas
                      #[0.5, "#FFFFFF"],  # Blanco para valores neutros (0)
                      [1, "#FF4B4B"]]  # Rojo para correlaciones positivas

        # Crear el heatmap con Plotly
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,  # Matriz de correlación
            x=corr.columns,  # Nombres de columnas en el eje X
            y=corr.index,  # Nombres de índices en el eje Y
            colorscale=colorscale,  # Escala de colores personalizada
            zmin=-1,  # Valor mínimo de correlación
            zmax=1  # Valor máximo de correlación
        ))
        
        return fig
        
        
    def show_donuts(self, cols=None):    
        return self._show_plot_2(False, "Distribución de Variables Categóricas",
                        "donut", cols)
        
    def show_pairplot(self):
        """Mostrar mapa de calor de correlación"""
        mi_paleta = DataManager.mi_paleta

        p = sns.pairplot(self.df, corner=True,
                 plot_kws={'color': mi_paleta['Black']},
                 diag_kws={'color': mi_paleta['Black']})

        p.fig.patch.set_facecolor(mi_paleta['Crimson'])
        for ax in p.axes.flatten():
            if ax is not None:
                ax.set_facecolor(mi_paleta['Crimson'])
                
                ax.tick_params(bottom=False, left=False)
                ax.tick_params(axis='y', colors=mi_paleta['Cream'],
                               labelsize=6)
                ax.tick_params(axis='x', colors=mi_paleta['Cream'],
                               labelsize=6)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(True)
                
                ax.xaxis.label.set_color(mi_paleta['Cream'])
                ax.yaxis.label.set_color(mi_paleta['Cream'])
                ax.xaxis.label.set_fontstyle('italic')
                ax.yaxis.label.set_fontstyle('italic')

        return p
    
    @staticmethod
    def hex_to_rgb(hex_color):
      """Converts a hexadecimal color code to an RGB tuple."""
      hex_color = hex_color.lstrip('#')
      return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def rgb_to_hex(rgb_color):
      """Converts an RGB tuple to a hexadecimal color code."""
      return '#%02x%02x%02x' % rgb_color
    
    @staticmethod
    def generate_gradient_colors(start_hex, end_hex, num_colors):
        """Generates a list of hexadecimal color codes between two given colors."""
        start_rgb = DataManager.hex_to_rgb(start_hex)
        end_rgb = DataManager.hex_to_rgb(end_hex)
    
        gradient_colors = []
        for i in range(num_colors):
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * (i / (num_colors - 1)))
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * (i / (num_colors - 1)))
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * (i / (num_colors - 1)))
            gradient_colors.append(DataManager.rgb_to_hex((r, g, b)))
        return gradient_colors

    def show_pairplot_st(self):
        
        df = self.df.copy()
        
        labels = self.tot_columns
        max_label_length = 20
        truncated_labels = [label if len(label) <= max_label_length
                            else label[:max_label_length] + "..."
                            for label in labels]
    
        
        dimensions = [{'label': truncated_label, 'values': df[label]}
                      for truncated_label, label in zip(truncated_labels,
                                                        labels)]
        
        fig = go.Figure(data=go.Splom(
                        dimensions=dimensions,
                        showupperhalf=False, # remove plots on diagonal
                        marker=dict(color="#FF4B4B", line_color='#31333F',
                                    line_width=0.5)
                        ))
        
        
        for i, label in enumerate(labels):
            fig.update_layout({
                f"xaxis{i+1}": dict(title_font=dict(size=12)),
                f"yaxis{i+1}": dict(title_font=dict(size=12))
            })
        
        fig.update_layout(
            autosize=True,
            width=1000,
            height=1000,
            hovermode='y',

       )

        return fig
        
    def show_facet_grid(self, rows, category, hue=None, type_p='histogram',
                        variable2=None):
        mi_paleta = DataManager.mi_paleta
        
        fig = []
        
        title =  True
        for (idx,variable) in enumerate(self.df[rows]):
            
            options = {'histogram': {'func':sns.histplot, 'args':[variable]},
                       'scatterplot': {'func':sns.scatterplot, 'args':[variable,
                                                                    variable2]},
                       'kdeplot': {'func':sns.kdeplot, 'args':[variable]}}
            
            g = sns.FacetGrid(self.df, col=category, hue=hue, palette=sns.dark_palette("#79C"),
                              height=4, aspect=1, dropna=False)
            g.fig.patch.set_facecolor(mi_paleta['Crimson'])
            
            g.map(options[type_p]['func'], *options[type_p]['args'])
        
            # Personalizar cada eje dentro del FacetGrid
            for ax in g.axes.flat:
                # Cambiar el color de fondo
                ax.set_facecolor(mi_paleta['Crimson'])
                
                # Cambiar el color de los ticks
                ax.tick_params(colors=mi_paleta['Cream'])
                
                # Cambiar el color de las etiquetas de los ejes
                ax.set_xlabel(ax.get_xlabel(), color=mi_paleta['Cream'])
                ax.set_ylabel('', color=mi_paleta['Cream'])
        
                if title:
                    ax.set_title(ax.get_title(), color=mi_paleta['Cream'])
                else:
                    ax.set_title('', color=mi_paleta['Cream'])
        
            title = False
            
            if hue:
                  g.add_legend(title=hue)
            
            fig.append(g)
            
        return fig
    
    def show_facet_grid_st(self, rows, category, hue=None, type_p='histogram',
                           variable2=None, traspuesta=False):
        figs = []
        
        data = self.df.copy()
        
        colors = ['#FF4B4B', '#31333F']
        color = [colors[0]]
        
        for variable in rows:
            
            if hue:
                ncolors = data[hue].nunique()
                color = DataManager.generate_gradient_colors(colors[0],
                                                                   colors[1],
                                                                   ncolors)
    
            if type_p == 'histogram':
                plot_axis = 'y' if traspuesta else 'x'
                plot_config = {'data_frame':self.df, plot_axis:variable,
                               'color':hue, 'facet_col':category,
                               'color_discrete_sequence':color}
                fig = px.histogram(**plot_config)
                

                if traspuesta:
                    fig.update_xaxes(title_text='', showgrid=True)
                else:
                    fig.update_yaxes(title_text='')
                
                
             
                
            elif type_p == 'scatterplot':
                fig = px.scatter(self.df, x=variable, y=variable2,
                                 color=hue, facet_col=category,
                                 color_discrete_sequence=color)
            
            figs.append(fig)

    
        # Actualiza la leyenda si se requiere
        if hue:
            fig.for_each_trace(lambda t: t.update(name=t.name + ' (hue)'))
    
        return figs
        
    @staticmethod
    def cat_2_num(col):
        if not is_numeric_dtype(col):
            new_col = col.astype('category').cat.codes
            return new_col
        return col
        
    # Función para pintar las reglas en 3D
    def rotable_3d(self, col1, col2, col3, cat_col):
        datos = self.df.copy()
        # Reseteo el índice de los datos originales
        datos.reset_index(inplace=True)
        colors = ['#FF4B4B', '#31333F']
        color = colors[0]
        
        if cat_col == None:
            datos = datos[[col1, col2, col3]].apply(self.cat_2_num)
            palette = None
            colorbar = None
        else:
            ncolors = datos[cat_col].nunique()
            datos = datos[[col1, col2, col3, cat_col]].apply(self.cat_2_num)
            color = datos[cat_col]
            palette = DataManager.generate_gradient_colors(colors[1],
                                                                 colors[0],
                                                                 ncolors)
            tickvals = [datos[cat_col].min(), 
                        datos[cat_col].max() - datos[cat_col].min(),
                        datos[cat_col].max()]
            colorbar = dict(  # Configuración de la barra de colores
                            title='Valores',  # Título de la barra de colores
                            titleside='right',  # Lado del título
                            tickvals=tickvals,  # Puedes añadir valores específicos para ticks aquí
                            ticktext=[],  # Puedes añadir texto para ticks aquí
                            )
    
        # Crear el scatter plot en 3D con Plotly
        fig = go.Figure(data=[go.Scatter3d(
            x=datos[col1],
            y=datos[col2],
            z=datos[col3],
            mode='markers',
            marker=dict(
                size=10,
                color=color,  # Color según el valor de la categoría
                colorscale=palette,  # Escala de color
                opacity=0.8,
                colorbar=colorbar
            ),
            text='<br>' + \
                 col1 + ": " + datos[col1].astype(str) + '<br>' + \
                 col2 + ": " + datos[col2].astype(str) + '<br>' + \
                 col3 + ": " + datos[col3].astype(str),
            hoverinfo='text'  # Mostrar texto en el menú emergente
        )])
    
        # Configuración del diseño del gráfico
        fig.update_layout(
            scene=dict(
                xaxis_title=col1,
                yaxis_title=col2,
                zaxis_title=col3,
            ),
            width=800,
            height=1200,
        )
        
        return fig
        
    @staticmethod
    def is_categorical(var_type):
        return var_type in ('object', 'category', 'string', 'datetime')
        
    def change_type(self, cols, new_type, new_categories=None, date_format=None):
        try:
            cols = cols if isinstance(cols, list) else [cols]
            if new_type == 'datetime':
                for col in cols:
                    self.df[col] = pd.to_datetime(self.df[col],
                                                  format=date_format,
                                                  errors='coerce')
            else:
                conversion = {col:new_type for col in cols}
                self.df = self.df.astype(conversion)
            
            if new_categories and len(cols) == 1 and new_type =='category':
                self.df[cols[0]] = pd.Categorical(self.df[cols[0]]
                                                  ).rename_categories(new_categories)
            self.split_df()
        except KeyError:
            raise KeyError(f"Tipo '{new_type}' no soportado para la columna '{cols}'.")
        except Exception as e:
                raise Exception(f"Error al convertir la columna '{cols}': {e}")      
        
    def change_name_col(self, col, new_name):
        self.df.rename(columns={col:new_name}, inplace=True)
        self.split_df()
        
    def save_df(self, path):
        self.df.to_csv(path, index=False, sep=';')
        
    def imput_vals(self, select, cols, fix_value=None):     
        imputations = {
        0: lambda col: self.df[col].fillna(self.df[col].mean(), 
                                           inplace=True),           # Media
        1: lambda col: self.df[col].fillna(self.df[col].median(), 
                                           inplace=True),           # Mediana
        2: lambda col: self.df[col].fillna(self.df[col].mode()[0], 
                                           inplace=True),           # Moda
        3: lambda col: self.df[col].fillna(method='ffill', 
                                           inplace=True),           # Valor anterior
        4: lambda col: self.df[col].fillna(method='bfill', 
                                           inplace=True),           # Valor posterior
        5: lambda col: self.df[col].fillna(fix_value,
                                           inplace=True),           # Valor fijo
        6: lambda col: self.df[col].fillna(pd.Timestamp(fix_value),
                                           inplace=True)            # Fecha fija
        }
        
        if select in imputations:
            cols = cols if isinstance(cols, list) else [cols]
            for col in cols:
                try:
                    imputations[select](col)
                except Exception as e:
                    raise Exception(f"Al imputar la columna '{col}': {e}")
        else:
            raise ValueError("Valor de select no válido. Debe ser 0, 1, 2, 3 o 4.")
            
    def drop_na(self, axis, cols=None, eval_method='any'):
        
        cols = cols if (isinstance(cols, (list, tuple)) and
                        not cols == None) else [cols]
        
        if axis==0 and cols:
            cols = cols if isinstance(cols, (list, tuple)) else [cols]
            self.df.dropna(axis=0, how=eval_method, subset=cols, inplace=True)
        
        if axis==1 and cols:
            
            if eval_method == 'any':
                self.df.drop(columns=[col for col in cols if self.df[col].isnull().any()],
                        inplace=True)
            if eval_method == 'all':
               self.df.drop(columns=[col for col in cols if self.df[col].isnull().all()],
                        inplace=True)
                
        self.tot_columns = self.df.columns.tolist()
        self.split_df()
        self.shape = self.df.shape
        self.n_rows = self.df.shape[0]
        self.n_columns = self.df.shape[1]
        
    def drop_duplicates(self, cols=None, keep='first'):
        
        df = self.show_duplicates(cols, keep)
        idx = df.index.tolist()
        
        if idx:
            self.df.drop(index=idx, inplace=True)
        self.tot_columns = self.df.columns.tolist()
        self.split_df()
        self.shape = self.df.shape
        self.n_rows = self.df.shape[0]
        self.n_columns = self.df.shape[1]
        
    def show_duplicates(self, cols=None, keep='first'):
        if cols == None:
            cols = self.tot_columns
            
        df = self.df[cols].copy()
        
        return self.df[df.duplicated(keep=keep)].copy()
    
    def show_nulls(self, cols=None, eval_method='any'):
        df = self.df.copy()
        if eval_method == 'any':
            df = df[df[cols].isnull().any(axis=1)]
        if eval_method == 'all':
            df = df[df[cols].isnull().all(axis=1)]
        return df.copy(), eval_method, cols
        
        
    def delete_columns(self, cols):
        if cols != None and len(cols) == 1:
            aux_cols = []
            aux_cols.append(cols)
            cols = aux_cols.copy()
        self.df.drop(columns=cols, inplace=True)
        self.tot_columns = self.df.columns.tolist()
        self.split_df()
        self.shape = self.df.shape
        self.n_rows = self.df.shape[0]
        self.n_columns = self.df.shape[1]
        
    def standarize_columns(self, cols):
        scaler = StandardScaler()
        self.df[cols] = scaler.fit_transform(self.df[cols])
        
    def normalize_columns(self, cols):
        normalizer = MinMaxScaler()
        self.df[cols] = normalizer.fit_transform(self.df[cols])
        
    def return_column_outliers(self, col):
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_condition = (self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))
        outliers = self.df[outlier_condition]
        
        return outliers[col], outlier_condition
        
    def delete_column_outliers(self, col):
        _, outlier_condition, = self.return_column_outliers(col)
        self.df = self.df[~outlier_condition]
        
    def winsorizing_column_outliers(self, col):
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        self.df[col] = np.where(self.df[col] > Q3 + 1.5 * IQR, Q3 + 1.5 * IQR,
                                self.df[col])
        self.df[col] = np.where(self.df[col] < Q1 - 1.5 * IQR, Q1 - 1.5 * IQR,
                                self.df[col])
    
    def return_string_columns(self):
        return self.df.select_dtypes(include=['object', 'string'])
        
    def string_strip(self, col):
        self.df[col] = self.df[col].str.strip()
        
    def string_upper(self, col):
        self.df[col] = self.df[col].str.upper()
    
    def string_lower(self, col):
        self.df[col] = self.df[col].str.lower()
        
    def string_replace(self, col, old_string, new_string):
        self.df[col] = self.df[col].str.replace(old_string, new_string)

if __name__ == "__main__":

    df = pd.read_csv('C:/Proyectos en GitHub/DataManagerApp/data/type_change.csv', sep=',')
    df['float_col']
    cols = ['float_col']
    # cols = ['integer_col']

    
    dm = DataManager(df.copy())
    dm.df[cols]
    dm.imput_vals(0, cols)
    dm.df[cols]