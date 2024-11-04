# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:11:06 2024

@author: Imanol
"""


import streamlit as st
from modules.streamlit.auxiliar import save_df

PAGE_CONFIG = {"page_title":"Guardar", 
               "layout":"wide"}

st.set_page_config(**PAGE_CONFIG)

def main():          
    if st.session_state.data_loaded:
        save_df(st.session_state.df)
    else:
        st.markdown("¡Todavia no se ha cargado ningún dataset!")
    
if __name__ == "__main__":
    main()