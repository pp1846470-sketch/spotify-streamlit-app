import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spotify Popularity Predictor",
    page_icon="🎵",
    layout="centered"
)

# ---------------- GREEN THEME CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #e8f5e9;
}
h1, h2, h3 {
    color: #1b5e20;
}
div[data-testid="stButton"] button {
    background-color: #2e7d32;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------- FEATURES (FIXED ORDER) ----------------
FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

# ---------------- TITLE ----------------
st.title("🎧 Spotify Song Popularity Predictor")
st.write("Predict whether a song will be **Popular or Not**")

# ---------------- INPUT UI ----------------
st.subheader("🎶 Enter Song Features")

inputs = {}
for feature in FEATURES:
    inputs[feature] = st.slider(
        feature.capitalize(),
        min_value=0.0,
        max_value=250.0 if feature == "tempo" else 1.0,
        value=0.5
    )

# ---------------- PREDICTION ----------------
if st.button("🎯 Predict Popularity"):
    input_df = pd.DataFrame([inputs])

    try:
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.success("🔥 This song is likely to be **POPULAR**!")
        else:
            st.warning("🎵 This song is likely to be **NOT popular**")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.write(e)
