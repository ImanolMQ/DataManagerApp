# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:25:45 2024

@author: Imanol
"""
from modules.streamlit.initial_exploration import DataExplorer
from modules.data_manager import DataManager
import streamlit as st


PAGE_CONFIG = {"page_title":"Paco el de la exploración", 
               "layout":"wide"}

st.set_page_config(**PAGE_CONFIG)


def main():
    if st.session_state.data_loaded:      
        dm = DataManager(st.session_state.df)
        ie = DataExplorer(dm)
        ie.render()
    else:
        st.markdown("¡Todavia no se ha cargado ningún dataset!")
        
if __name__ == "__main__":
    main()