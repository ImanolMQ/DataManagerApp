# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:10:40 2024

@author: Imanol
"""

import streamlit as st
from modules.data_manager import DataManager
from modules.streamlit.display_graphs import GraphsDisplayer

PAGE_CONFIG = {"page_title":"Visualización", 
               "layout":"wide"}

st.set_page_config(**PAGE_CONFIG)


def main():     
    if st.session_state.data_loaded:
        dm = DataManager(st.session_state.df)
        gd = GraphsDisplayer(dm)
        gd.show_data_exploration_graphs()
    else:
        st.markdown("¡Todavia no se ha cargado ningún dataset!")
        
if __name__ == "__main__":
    main()