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

# ✅ EXACT FEATURES USED DURING TRAINING
FEATURES = [
    "artist_popularity",
    "artist_followers",
    "track_duration_min",
    "album_type",
    "explicit"
]

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
# SONG POPULARITY PREDICTION (FIXED)
# =================================================
elif page == "Song Popularity Prediction":
    st.title("🔥 Song Popularity Prediction")

    col1, col2 = st.columns(2)

    with col1:
        artist_popularity = st.slider("Artist Popularity", 0, 100, 50)
        artist_followers = st.number_input("Artist Followers", min_value=0, value=50000)
        track_duration_min = st.slider("Track Duration (minutes)", 1.0, 10.0, 3.5)

    with col2:
        album_type = st.selectbox("Album Type", ["album", "single"])
        explicit = st.selectbox("Explicit Content", ["No", "Yes"])

    if st.button("Predict Song Success"):
        # Create input DataFrame EXACTLY like training
        input_df = pd.DataFrame([{
            "artist_popularity": artist_popularity,
            "artist_followers": artist_followers,
            "track_duration_min": track_duration_min,
            "album_type": 1 if album_type == "single" else 0,
            "explicit": 1 if explicit == "Yes" else 0
        }])[FEATURES]

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.success("🎯 HIT SONG")
        else:
            st.error("❌ NOT A HIT")

        st.progress(prob)
        st.write(f"**Probability of Success:** {prob:.2f}")

        if prob > 0.7:
            st.write("🟢 High Confidence")
        elif prob > 0.4:
            st.write("🟡 Medium Confidence")
        else:
            st.write("🔴 Low Confidence")

# =================================================
# ARTIST & GENRE ANALYSIS
# =================================================
elif page == "Artist & Genre Analysis":
    st.title("🎤 Artist & Genre Analysis")

    genre_pop = df.groupby("artist_genres")["track_popularity"].mean().sort_values(ascending=False).head(10)
    st.subheader("Top Genres by Popularity")
    st.bar_chart(genre_pop)

    st.subheader("Genre Distribution")
    genre_count = df["artist_genres"].value_counts().head(6)
    fig, ax = plt.subplots()
    ax.pie(genre_count, labels=genre_count.index, autopct="%1.1f%%")
    st.pyplot(fig)

# =================================================
# ALBUM INSIGHTS
# =================================================
elif page == "Album Insights":
    st.title("💿 Album Insights")

    fig, ax = plt.subplots()
    sns.boxplot(x="album_type", y="track_popularity", data=df, ax=ax)
    st.subheader("Album Type vs Popularity")
    st.pyplot(fig)

    st.subheader("Popularity Over Years")
    year_trend = df.groupby("album_release_year")["track_popularity"].mean()
    st.area_chart(year_trend)

# =================================================
# MODEL PERFORMANCE
# =================================================
elif page == "Model Performance":
    st.title("🧠 Model Performance")

    X = df[FEATURES].copy()
    y = df["popular"]

    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
    st.pyplot(fig)

    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.write("Precision:", precision_score(y, y_pred))
    st.write("Recall:", recall_score(y, y_pred))
    st.write("F1 Score:", f1_score(y, y_pred))
