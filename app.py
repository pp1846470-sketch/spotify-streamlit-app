import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Spotify Song Success Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# GREEN THEME CSS
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
h1, h2, h3 {
    color: #1DB954;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
}
.center {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA & MODEL
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("spotify_preprocessed_dataset.csv")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

FEATURES = list(model.feature_names_in_)  # 🔥 KEY FIX

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Song Popularity Prediction",
        "Artist & Genre Analysis",
        "Album Insights",
        "Model Performance"
    ]
)

# =================================================
# HOME
# =================================================
if page == "Home":
    st.markdown("""
    <div class="center card">
        <h1>🎧 Spotify Song Success Intelligence Platform</h1>
        <h3>Machine Learning Based Hit Prediction</h3>
        <br>
        <p>Created by <b>Pooja Parmar</b></p>
    </div>
    """, unsafe_allow_html=True)

# =================================================
# PREDICTION PAGE (FIXED)
# =======
