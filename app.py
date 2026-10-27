import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

# Setup page config first thing
st.set_page_config(
    page_title="E-Learning Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# Get the base directory where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Define straight paths directly in the root folder
MODEL_PATH = BASE_DIR / "extra_trees_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.pkl"

@st.cache_resource
def load_model():
    # Load files directly using the clean path variables
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

# Load the model and preprocessor
try:
    model, preprocessor = load_model()
    st.success("Model and preprocessor loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")

# Rest of your app interface code goes here...
st.title("🎓 E-Learning Dropout Prediction App")