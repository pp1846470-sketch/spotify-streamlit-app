import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Spotify Song Success Predictor", layout="wide")

# ---------------------------
# Load Data & Model
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("spotify_preprocessed_dataset.csv")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("🎵 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Song Popularity Prediction", "Artist & Genre Analysis", "Album Insights", "Song Explorer"]
)

# ---------------------------
# HOME PAGE
# ---------------------------
if page == "Home":
    st.title("🎧 Spotify Song Success Analyzer")
    st.markdown("""
    ### 📌 Project Objective
    This web application predicts whether a Spotify song will become **popular (Hit)** or **not popular**
    using **Logistic Regression**.

    ### 🔍 Features
    - Binary classification (Hit / Not Hit)
    - Artist & genre insights
    - Album trend analysis
    - Interactive Streamlit dashboard

    ### 🧠 Model Used
    - Logistic Regression
    - Supervised Machine Learning
    """)

# ---------------------------
# PREDICTION PAGE
# ---------------------------
elif page == "Song Popularity Prediction":
    st.title("🔥 Song Popularity Prediction (Hit / Not Hit)")

    col1, col2 = st.columns(2)

    with col1:
        artist_popularity = st.slider("Artist Popularity", 0, 100, 50)
        artist_followers = st.number_input("Artist Followers", min_value=0, value=100000)
        track_duration = st.slider("Track Duration (minutes)", 1.0, 10.0, 3.5)

    with col2:
        album_type = st.selectbox("Album Type", ["album", "single"])
        explicit = st.selectbox("Explicit Content", ["No", "Yes"])

    album_type_encoded = 1 if album_type == "single" else 0
    explicit_encoded = 1 if explicit == "Yes" else 0

    input_data = np.array([[artist_popularity,
                            artist_followers,
                            track_duration,
                            album_type_encoded,
                            explicit_encoded]])

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"🔥 HIT SONG (Probability: {probability:.2f})")
        else:
            st.warning(f"❌ NOT A HIT (Probability: {probability:.2f})")

# ---------------------------
# ARTIST & GENRE ANALYSIS
# ---------------------------
elif page == "Artist & Genre Analysis":
    st.title("🎤 Artist & Genre Analysis")

    top_artists = df.groupby("artist_name")["track_popularity"].mean().sort_values(ascending=False).head(10)

    st.subheader("Top 10 Artists by Average Popularity")
    st.bar_chart(top_artists)

    st.subheader("Followers vs Track Popularity")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df["artist_followers"],
        y=df["track_popularity"],
        ax=ax
    )
    st.pyplot(fig)

# ---------------------------
# ALBUM INSIGHTS
# ---------------------------
elif page == "Album Insights":
    st.title("💿 Album Insights")

    album_popularity = df.groupby("album_type")["track_popularity"].mean()

    st.subheader("Album Type vs Popularity")
    st.bar_chart(album_popularity)

    year_trend = df.groupby("album_release_year")["track_popularity"].mean()

    st.subheader("Popularity Trend Over Years")
    st.line_chart(year_trend)

# ---------------------------
# SONG EXPLORER
# ---------------------------
elif page == "Song Explorer":
    st.title("🎶 Song Explorer")

    song = st.selectbox("Select a Song", df["track_name"].unique())
    song_data = df[df["track_name"] == song].iloc[0]

    st.write("### Song Details")
    st.write(f"**Artist:** {song_data['artist_name']}")
    st.write(f"**Popularity:** {song_data['track_popularity']}")
    st.write(f"**Explicit:** {song_data['explicit']}")
    st.write(f"**Album Type:** {song_data['album_type']}")
    st.write(f"**Release Year:** {song_data['album_release_year']}")
