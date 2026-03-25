import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    with open("model_pipeline.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved

saved = load_model()

model = saved["model"]
scaler = saved["scaler"]
features = saved["features"]
threshold = saved.get("threshold", 0.5)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv("spotify_preprocessed_dataset.csv")

df = load_data()

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Spotify Hit Predictor", layout="wide")

# ------------------ TITLE ------------------
st.title("🎵 Spotify Song Hit Prediction App")
st.markdown("Using Logistic Regression Model")

# ------------------ SIDEBAR ------------------
menu = st.sidebar.radio("Menu", ["Home", "Prediction", "Analysis", "About"])

# ------------------ HOME ------------------
if menu == "Home":
    st.subheader("📌 Project Overview")
    st.write("""
    This project predicts whether a song will be a HIT or NOT based on its audio features.
    
    🔹 Algorithm Used: Logistic Regression  
    🔹 Input: Audio features (danceability, energy, tempo, etc.)  
    🔹 Output: Hit or Not Hit  
    """)

    st.dataframe(df.head())

# ------------------ PREDICTION ------------------
elif menu == "Prediction":
    st.subheader("🎯 Make Prediction")

    input_data = {}

    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(features):
        if i % 3 == 0:
            input_data[feature] = col1.number_input(feature, value=0.0)
        elif i % 3 == 1:
            input_data[feature] = col2.number_input(feature, value=0.0)
        else:
            input_data[feature] = col3.number_input(feature, value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Prediction
        prob = model.predict_proba(input_scaled)[:, 1][0]
        pred = 1 if prob > threshold else 0

        st.subheader("📊 Result")

        st.write(f"Prediction Probability: **{round(prob, 3)}**")

        if pred == 1:
            st.success("🔥 This song is likely to be a HIT!")
        else:
            st.error("❌ This song may NOT be a hit.")

# ------------------ ANALYSIS ------------------
elif menu == "Analysis":
    st.subheader("📈 Data Analysis")

    st.write("### Popularity Distribution")
    fig1 = px.histogram(df, x="popularity")
    st.plotly_chart(fig1, use_container_width=True)

    st.write("### Correlation Heatmap")
    corr = df.corr()
    fig2 = px.imshow(corr)
    st.plotly_chart(fig2, use_container_width=True)

# ------------------ ABOUT ------------------
elif menu == "About":
    st.subheader("ℹ️ About")

    st.write("""
    🎵 Spotify Song Success Prediction Project
    
    🔹 Algorithm: Logistic Regression  
    🔹 Built with: Streamlit, Scikit-learn, Plotly  
    🔹 Developed for Academic Purpose  
    """)
