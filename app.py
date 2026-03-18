import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Music Dashboard", layout="wide")

# Custom CSS for dark UI
st.markdown("""
    <style>
    body {
        background-color: #0e1a2b;
        color: white;
    }
    .card {
        background-color: #162447;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("spotify_songs.csv")

st.title("🎵 Music Popularity Dashboard")

# ---- TOP CARDS ----
col1, col2, col3, col4 = st.columns(4)

col1.markdown(f'<div class="card"><h3>Total Songs</h3><h2>{len(df)}</h2></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="card"><h3>Avg Popularity</h3><h2>{round(df["popularity"].mean(),2)}</h2></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="card"><h3>Avg Energy</h3><h2>{round(df["energy"].mean(),2)}</h2></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="card"><h3>Avg Danceability</h3><h2>{round(df["danceability"].mean(),2)}</h2></div>', unsafe_allow_html=True)

# ---- MAIN LAYOUT ----
col1, col2 = st.columns(2)

# Popularity Distribution
with col1:
    st.subheader("Popularity Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['popularity'], bins=20)
    st.pyplot(fig)

# Energy vs Popularity
with col2:
    st.subheader("Energy vs Popularity")
    fig, ax = plt.subplots()
    ax.scatter(df['energy'], df['popularity'])
    ax.set_xlabel("Energy")
    ax.set_ylabel("Popularity")
    st.pyplot(fig)

# ---- SECOND ROW ----
col3, col4 = st.columns(2)

# Danceability vs Popularity
with col3:
    st.subheader("Danceability vs Popularity")
    fig, ax = plt.subplots()
    ax.scatter(df['danceability'], df['popularity'])
    st.pyplot(fig)

# Top 10 Songs
with col4:
    st.subheader("Top 10 Songs")
    top10 = df.sort_values(by="popularity", ascending=False).head(10)
    st.dataframe(top10[['track_name', 'artist_name', 'popularity']])

# ---- BOTTOM ----
st.subheader("Feature Correlation")
corr = df.corr(numeric_only=True)

fig, ax = plt.subplots()
cax = ax.matshow(corr)
fig.colorbar(cax)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)

st.pyplot(fig)
