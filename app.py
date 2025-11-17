import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

#definir working directory
wd = os.path.dirname(os.path.abspath(__file__))

st.title('La durée d\'hospitalisation prédite')


number = st.number_input(
    "Quel est ton âge ?", value=None, placeholder="Type a number..."
)

st.write("Ton âge est de ", number)