import streamlit as st
import pandas as pd
import numpy as np
import pickle

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

# ------------------ UI ------------------
st.title("🎵 Spotify Song Success Prediction")

# Inputs (same order as features)
input_data = {}

for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("Predict"):
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict_proba(input_scaled)[:, 1][0]
    pred = 1 if prob > threshold else 0

    st.write(f"Prediction Probability: {round(prob, 3)}")

    if pred == 1:
        st.success("🔥 Hit Song!")
    else:
        st.error("❌ Not a Hit")
