# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:35:55 2024

@author: Imanol
"""

import streamlit as st
from modules.data_manager import DataManager
from modules.streamlit.data_cleaning import DataCleaner


PAGE_CONFIG = {"page_title":"Limpieza de datos", 
               "layout":"wide"}

st.set_page_config(**PAGE_CONFIG)

def main():
    if st.session_state.data_loaded:
        dm = DataManager(st.session_state.df)
        dc = DataCleaner(dm)
        st.session_state.df = dc.render()
    else:
        st.markdown("¡Todavia no se ha cargado ningún dataset!")
    
if __name__ == "__main__":
    main()