import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu
import plotly.express as px

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    with open("logistic_model.pkl", "rb") as f:
        logistic_model = pickle.load(f)

    with open("random_forest_model.pkl", "rb") as f:
        rf_model = pickle.load(f)

    return logistic_model, rf_model

logistic_model, rf_model = load_models()

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv("spotify_preprocessed_dataset.csv")

df = load_data()

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Spotify Predictor", layout="wide")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Home", "Prediction", "Analysis", "About"],
        icons=["house", "music-note", "bar-chart", "info-circle"],
        default_index=0
    )

# ------------------ HOME ------------------
if selected == "Home":
    st.title("🎵 Spotify Song Success Prediction")
    st.write("Compare Logistic Regression vs Random Forest 🚀")
    st.dataframe(df.head())

# ------------------ PREDICTION ------------------
elif selected == "Prediction":
    st.title("🎯 Song Success Prediction")

    # 🔥 MODEL SELECTION
    model_choice = st.selectbox(
        "Select Algorithm",
        ["Logistic Regression", "Random Forest"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        energy = st.slider("Energy", 0.0, 1.0, 0.5)
        loudness = st.slider("Loudness", -60.0, 0.0, -5.0)

    with col2:
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)

    with col3:
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
        valence = st.slider("Valence", 0.0, 1.0, 0.5)
        tempo = st.slider("Tempo", 50.0, 200.0, 120.0)

    if st.button("Predict"):
        input_data = np.array([[danceability, energy, loudness,
                                speechiness, acousticness, instrumentalness,
                                liveness, valence, tempo]])

        # 🔥 MODEL LOGIC
        if model_choice == "Logistic Regression":
            prediction = logistic_model.predict(input_data)[0]
        else:
            prediction = rf_model.predict(input_data)[0]

        # 🔥 OUTPUT
        st.subheader(f"Model Used: {model_choice}")

        if prediction == 1:
            st.success("🔥 This song is likely to be a HIT!")
        else:
            st.error("❌ This song may NOT be a hit.")

# ------------------ ANALYSIS ------------------
elif selected == "Analysis":
    st.title("📊 Data Analysis")

    fig1 = px.histogram(df, x="popularity")
    st.plotly_chart(fig1, use_container_width=True)

    corr = df.corr()
    fig2 = px.imshow(corr)
    st.plotly_chart(fig2, use_container_width=True)

# ------------------ ABOUT ------------------
elif selected == "About":
    st.title("ℹ️ About Project")

    st.write("""
    🔹 Algorithms Used:
    - Logistic Regression
    - Random Forest Classifier

    🔹 Features:
    danceability, energy, loudness, tempo, etc.

    🔹 Built with:
    Streamlit, Scikit-learn, Plotly
    """)
