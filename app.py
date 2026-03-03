import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Spotify Song Success Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
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

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_preprocessed_dataset.csv")
    return df

# ===============================
# LOAD MODEL PIPELINE
# ===============================
@st.cache_resource
def load_model():
    import os
    st.write("Files in directory:", os.listdir())
    saved = joblib.load("model_pipeline.pkl")
    st.write("Loaded object type:", type(saved))
    return saved
    
df = load_data()
saved = load_model()

model = saved["model"]
scaler = saved["scaler"]
features = saved["features"]
threshold = saved["threshold"]

# ===============================
# SIDEBAR
# ===============================
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

# ===============================
# HOME
# ===============================
if page == "Home":
    st.markdown("""
    <div class="center card">
        <h1>🎧 Spotify Song Success Intelligence Platform</h1>
        <h3>Machine Learning Based Hit Prediction</h3>
        <br>
        <p>Created by <b>Pooja Parmar</b></p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Song Popularity Prediction":

    st.title("Song Popularity Prediction")

    col1, col2 = st.columns(2)

    with col1:
        artist_popularity = st.slider("Artist Popularity", 0, 100, 50)
        artist_followers = st.number_input("Artist Followers", min_value=0, value=50000)
        track_duration = st.slider("Track Duration (minutes)", 1.0, 10.0, 3.5)

    with col2:
        album_type = st.selectbox("Album Type", ["album", "single"])
        explicit = st.selectbox("Explicit Content", ["No", "Yes"])
        album_total_tracks = st.number_input("Album Total Tracks", 1, 50, 10)

    # Encoding
    album_encoded = 1 if album_type == "single" else 0
    explicit_encoded = 1 if explicit == "Yes" else 0

    # Create input dictionary (MATCH TRAINING)
    input_dict = {
        "explicit": explicit_encoded,
        "artist_popularity": artist_popularity,
        "artist_followers": artist_followers,
        "album_total_tracks": album_total_tracks,
        "track_duration_min": track_duration,
        "album_type": album_encoded
    }

    # Convert to DataFrame
    X_input = pd.DataFrame([input_dict])

    # Ensure correct order
    X_input = X_input[features]

    # Scale input
    X_scaled = scaler.transform(X_input)

    if st.button("Predict Song Success"):

        prob = model.predict_proba(X_scaled)[0][1]
        pred = 1 if prob > threshold else 0

        st.subheader("Prediction Result")

        if pred == 1:
            st.success("HIT SONG")
        else:
            st.error("NOT A HIT")

        st.progress(float(prob))
        st.write(f"**Probability of Success:** {prob:.2f}")

# ===============================
# ANALYSIS
# ===============================
elif page == "Artist & Genre Analysis":

    st.title("Artist & Genre Analysis")

    genre_pop = df.groupby("artist_genres")["track_popularity"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(genre_pop)

# ===============================
# ALBUM
# ===============================
elif page == "Album Insights":

    st.title("Album Insights")

    fig, ax = plt.subplots()
    sns.boxplot(x="album_type", y="track_popularity", data=df, ax=ax)
    st.pyplot(fig)

# ===============================
# MODEL PERFORMANCE
# ===============================
elif page == "Model Performance":

    st.title("Model Performance")

    X = df[features].copy()

    X["album_type"] = (X["album_type"] == "single").astype(int)
    X["explicit"] = (X["explicit"] == True).astype(int)

    X_scaled = scaler.transform(X)
    y = df["popular"]

    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.write("Precision:", precision_score(y, y_pred))
    st.write("Recall:", recall_score(y, y_pred))
    st.write("F1 Score:", f1_score(y, y_pred))
