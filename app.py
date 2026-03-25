import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import warnings
warnings.filterwarnings("ignore")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Spotify Hit Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #060608; color: #fff; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}

[data-testid="stSidebar"] {
    background: #0d0d10;
    border-right: 1px solid rgba(29,185,84,0.3);
}
[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
[data-testid="stSidebar"] .stRadio label {
    color: #888 !important; font-size: 14px !important;
    padding: 10px 16px !important; border-radius: 10px !important;
    transition: all 0.2s !important; border: 1px solid transparent !important;
    display: block;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #fff !important; background: rgba(29,185,84,0.1) !important;
    border-color: rgba(29,185,84,0.2) !important;
}

.card {
    background: linear-gradient(145deg, #111114, #0e0e12);
    border: 1px solid #1e1e24; border-radius: 20px;
    padding: 22px 20px; text-align: center;
    transition: all 0.3s; position: relative; overflow: hidden;
}
.card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: linear-gradient(90deg, #1db954, #1ed760);
    opacity: 0; transition: opacity 0.3s;
}
.card:hover { transform: translateY(-5px); border-color: rgba(29,185,84,0.3); }
.card:hover::before { opacity: 1; }
.card-value { font-family: 'Syne', sans-serif; font-size: 34px; font-weight: 800; color: #1db954; line-height: 1; }
.card-label { font-size: 11px; color: #555; margin-top: 6px; text-transform: uppercase; letter-spacing: 1.5px; }
.card-sub { font-size: 12px; color: #888; margin-top: 4px; }

.hit-banner {
    background: linear-gradient(135deg, #0f3d1f, #145c2a);
    border: 1px solid #1db954; border-radius: 20px; padding: 30px;
    text-align: center; margin: 16px 0;
    box-shadow: 0 0 40px rgba(29,185,84,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
}
.miss-banner {
    background: linear-gradient(135deg, #3d0f1f, #5c1428);
    border: 1px solid #e8175d; border-radius: 20px; padding: 30px;
    text-align: center; margin: 16px 0;
    box-shadow: 0 0 40px rgba(232,23,93,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
}
.banner-icon { font-size: 52px; margin-bottom: 8px; }
.banner-title { font-family: 'Syne', sans-serif; font-size: 30px; font-weight: 800; color: #fff; margin: 0; }
.banner-sub { font-size: 15px; color: rgba(255,255,255,0.6); margin-top: 8px; }

.sec-head {
    font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 700; color: #fff;
    display: flex; align-items: center; gap: 10px; margin: 28px 0 16px 0;
}
.sec-head::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #1e1e24, transparent);
}

.stNumberInput input {
    background: #111114 !important; border: 1px solid #1e1e24 !important;
    color: #fff !important; border-radius: 10px !important; font-size: 14px !important;
}
.stNumberInput input:focus {
    border-color: #1db954 !important;
    box-shadow: 0 0 0 3px rgba(29,185,84,0.12) !important;
}
.stSelectbox > div > div {
    background: #111114 !important; border: 1px solid #1e1e24 !important;
    border-radius: 10px !important; color: #fff !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1db954, #17a045) !important;
    color: #000 !important; border: none !important; border-radius: 50px !important;
    padding: 13px 40px !important; font-size: 15px !important; font-weight: 700 !important;
    transition: all 0.25s !important; width: 100% !important;
}
.stButton > button:hover {
    transform: scale(1.04) !important;
    box-shadow: 0 8px 30px rgba(29,185,84,0.35) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0d0d10 !important; border-radius: 12px !important;
    padding: 4px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #666 !important;
    border-radius: 8px !important; font-size: 13px !important; padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(29,185,84,0.15) !important; color: #1db954 !important;
}

.pill {
    display: inline-block; background: rgba(29,185,84,0.1);
    border: 1px solid rgba(29,185,84,0.25); border-radius: 50px;
    padding: 5px 14px; font-size: 12px; color: #1db954; margin: 3px;
}
.ibox {
    background: #0d0d10; border: 1px solid #1e1e24; border-radius: 14px;
    padding: 18px 20px; margin: 10px 0; font-size: 14px; color: #888; line-height: 1.7;
}
.ibox b, .ibox strong { color: #1db954; }

h1 {
    font-family: 'Syne', sans-serif !important;
    background: linear-gradient(90deg, #1db954 0%, #1ed760 50%, #fff 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important; font-size: 40px !important; line-height: 1.1 !important;
}
h2, h3 { font-family: 'Syne', sans-serif !important; color: #fff !important; }
label { color: #777 !important; font-size: 12px !important; }
p { color: #aaa; }

.dot-green { width:8px;height:8px;background:#1db954;border-radius:50%;display:inline-block;margin-right:6px;box-shadow:0 0 6px #1db954; }
.dot-red   { width:8px;height:8px;background:#e8175d;border-radius:50%;display:inline-block;margin-right:6px;box-shadow:0 0 6px #e8175d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  AUTO-DETECT PKL FILE
# ─────────────────────────────────────────────
def find_pkl_file():
    for pattern in ["model_pipeline*.pkl", "model_pipeline.pkl", "*.pkl"]:
        files = glob.glob(pattern)
        if files:
            model_files = [f for f in files if "model" in f.lower()]
            return model_files[0] if model_files else files[0]
    return None


# ─────────────────────────────────────────────
#  LOAD MODEL & DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    pkl_path = find_pkl_file()
    if pkl_path is None:
        raise FileNotFoundError("No .pkl model file found in the project directory.")
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    return saved, pkl_path


@st.cache_data
def load_data():
    for pattern in ["spotify*.csv", "*.csv"]:
        files = glob.glob(pattern)
        if files:
            return pd.read_csv(files[0]), files[0]
    raise FileNotFoundError("No CSV dataset found.")


try:
    saved, pkl_path = load_model()
    model     = saved["model"]
    scaler    = saved["scaler"]
    features  = saved["features"]
    threshold = saved.get("threshold", 0.5)
    model_loaded = True
    pkl_name  = os.path.basename(pkl_path)
except Exception as e:
    model_loaded = False
    model_error  = str(e)
    pkl_name     = "not found"
    features     = []
    threshold    = 0.5

try:
    df, csv_name = load_data()
    data_loaded  = True
    HIT_COL    = next((c for c in ["hit","target","label","is_hit"] if c in df.columns), None)
    ARTIST_COL = next((c for c in ["artist","artist_name","artists","artist_names","performer"] if c in df.columns), None)
    TRACK_COL  = next((c for c in ["track","track_name","song","title","name"] if c in df.columns), None)
    NUM_COLS   = df.select_dtypes(include=np.number).columns.tolist()
except Exception as e:
    data_loaded  = False
    df           = pd.DataFrame()
    HIT_COL = ARTIST_COL = TRACK_COL = None
    NUM_COLS = []


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:24px 0 8px'>
        <div style='font-size:48px'>🎵</div>
        <div style='font-family:Syne,sans-serif;font-size:16px;font-weight:800;
                    background:linear-gradient(90deg,#1db954,#1ed760);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-top:6px'>
            Spotify Hit Predictor</div>
        <div style='font-size:11px;color:#444;margin-top:2px'>Final Year ML Project</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    menu = st.radio("Navigation",
        ["🏠  Home","🎯  Prediction","📊  Analysis",
         "🎤  Artist Insights","🔬  Model Insights","ℹ️  About"],
        label_visibility="collapsed")

    st.divider()

    if model_loaded:
        st.markdown(f'<div class="ibox"><span class="dot-green"></span><b>Model Loaded</b><br><span style="font-size:11px;color:#555">{pkl_name}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ibox" style="border-color:rgba(232,23,93,0.3)"><span class="dot-red"></span><b style="color:#e8175d">Model Error</b><br><span style="font-size:11px;color:#555">{model_error[:55]}</span></div>', unsafe_allow_html=True)

    if data_loaded:
        st.markdown(f'<div class="ibox"><span class="dot-green"></span><b>Dataset Ready</b><br><span style="font-size:11px;color:#555">{len(df):,} songs · {len(df.columns)} cols</span></div>', unsafe_allow_html=True)
        if ARTIST_COL:
            st.markdown(f'<div class="ibox"><span class="dot-green"></span><b>Artists Found</b><br><span style="font-size:11px;color:#555">{df[ARTIST_COL].nunique():,} unique artists</span></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  🏠 HOME
# ═══════════════════════════════════════════════════════════
if menu == "🏠  Home":
    st.title("Spotify Hit Predictor")
    st.markdown("##### ML-powered chart prediction using Spotify's audio DNA")
    st.divider()

    if data_loaded and len(df) > 0:
        c1, c2, c3, c4, c5 = st.columns(5)
        hit_rate  = df[HIT_COL].mean() * 100 if HIT_COL else 0
        avg_pop   = df["popularity"].mean() if "popularity" in df.columns else 0
        n_artists = df[ARTIST_COL].nunique() if ARTIST_COL else 0

        with c1: st.markdown(f'<div class="card"><div class="card-value">{len(df):,}</div><div class="card-label">Total Songs</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="card"><div class="card-value">{n_artists:,}</div><div class="card-label">Artists</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="card"><div class="card-value">{hit_rate:.1f}%</div><div class="card-label">Hit Rate</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="card"><div class="card-value">{avg_pop:.0f}</div><div class="card-label">Avg Popularity</div></div>', unsafe_allow_html=True)
        with c5: st.markdown(f'<div class="card"><div class="card-value">{len(NUM_COLS)}</div><div class="card-label">Features</div></div>', unsafe_allow_html=True)

    st.divider()
    cl, cr = st.columns([1.3, 1])

    with cl:
        st.markdown('<div class="sec-head">About This Project</div>', unsafe_allow_html=True)
        st.markdown("""<div class="ibox">
        Uses <b>Logistic Regression</b> trained on Spotify audio features to predict chart hits.<br><br>
        <b>🎛 Algorithm:</b> Logistic Regression &nbsp;|&nbsp; <b>📐 Scaler:</b> Standard Scaler<br>
        <b>🎯 Output:</b> Hit Probability (0–100%) &nbsp;|&nbsp; <b>🛠 Stack:</b> Streamlit · Scikit-learn · Plotly
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head">How It Works</div>', unsafe_allow_html=True)
        for n, t, d in [
            ("01","Audio Input","Enter 13+ Spotify audio features"),
            ("02","Normalise","Standard Scaler transforms raw values"),
            ("03","Inference","Logistic Regression scores the song"),
            ("04","Decision",f"Hit if score ≥ {threshold:.0%} threshold"),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:14px;align-items:flex-start;padding:12px 14px;
                        background:#0d0d10;border:1px solid #1a1a20;border-radius:12px;margin:6px 0">
                <div style="font-family:Syne,sans-serif;font-size:11px;font-weight:800;
                            color:#1db954;opacity:0.6;min-width:28px;padding-top:2px">{n}</div>
                <div>
                    <div style="font-weight:600;color:#fff;font-size:14px">{t}</div>
                    <div style="font-size:12px;color:#555;margin-top:2px">{d}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with cr:
        if data_loaded and HIT_COL:
            hits    = int(df[HIT_COL].sum())
            nonhits = len(df) - hits
            fig_d = go.Figure(go.Pie(
                labels=["Hit 🔥","Not Hit ❌"], values=[hits, nonhits], hole=0.65,
                marker=dict(colors=["#1db954","#e8175d"], line=dict(color="#060608",width=3)),
                textinfo="label+percent", textfont=dict(color="white",size=12),
                hovertemplate="<b>%{label}</b><br>%{value:,} songs<extra></extra>"
            ))
            fig_d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False, font=dict(color="white"),
                margin=dict(t=20,b=20,l=20,r=20),
                annotations=[dict(text=f"<b>{hits:,}</b><br>Hits", x=0.5, y=0.5,
                                  font=dict(size=18,color="white"), showarrow=False)]
            )
            st.plotly_chart(fig_d, use_container_width=True)

        if data_loaded and ARTIST_COL and HIT_COL:
            top5 = (df[df[HIT_COL]==1].groupby(ARTIST_COL)
                    .size().reset_index(name="hits")
                    .sort_values("hits", ascending=False).head(5))
            st.markdown('<div class="sec-head">Top Hit Artists</div>', unsafe_allow_html=True)
            for _, row in top5.iterrows():
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:10px 14px;background:#0d0d10;border:1px solid #1a1a20;
                            border-radius:10px;margin:4px 0">
                    <span style="color:#fff;font-size:13px;font-weight:500">{row[ARTIST_COL]}</span>
                    <span class="pill">{row['hits']} hits</span>
                </div>""", unsafe_allow_html=True)

    if data_loaded:
        st.divider()
        st.markdown('<div class="sec-head">Dataset Preview</div>', unsafe_allow_html=True)
        preview = df.head(10)
        st.dataframe(
            preview.style.background_gradient(
                cmap="Greens",
                subset=[c for c in NUM_COLS if c in preview.columns]
            ),
            use_container_width=True, height=320
        )


# ═══════════════════════════════════════════════════════════
#  🎯 PREDICTION
# ═══════════════════════════════════════════════════════════
elif menu == "🎯  Prediction":
    st.title("Hit Predictor")
    st.markdown("##### Enter audio features — get an instant hit probability score")
    st.divider()

    if not model_loaded:
        st.error(f"❌ Model file not found. Put your `.pkl` file in the same folder as `app.py`.\n\nError: `{model_error}`")
        st.stop()

    HELP = {
        "danceability":"How suitable for dancing · 0.0 – 1.0",
        "energy":"Intensity & power · 0.0 – 1.0",
        "key":"Musical key · 0 (C) to 11 (B)",
        "loudness":"Overall loudness in dB · −60 to 0",
        "mode":"Major = 1, Minor = 0",
        "speechiness":"Spoken word presence · 0.0 – 1.0",
        "acousticness":"Acoustic confidence · 0.0 – 1.0",
        "instrumentalness":"Vocal absence · 0.0 – 1.0",
        "liveness":"Live audience presence · 0.0 – 1.0",
        "valence":"Musical positiveness · 0.0 – 1.0",
        "tempo":"Beats per minute",
        "duration_ms":"Duration in milliseconds",
        "time_signature":"Beats per bar · 3 – 7",
        "popularity":"Spotify popularity score · 0 – 100",
    }

    input_data = {}
    cols = st.columns(3)
    for i, feat in enumerate(features):
        input_data[feat] = cols[i % 3].number_input(
            feat.replace("_"," ").title(), value=0.0,
            format="%.4f", help=HELP.get(feat,"")
        )

    st.markdown('<div class="sec-head">🕸 Audio Feature Radar</div>', unsafe_allow_html=True)
    radar_feats = [f for f in features if f in
                   ["danceability","energy","speechiness","acousticness",
                    "instrumentalness","liveness","valence"]]
    if radar_feats:
        rvals = [input_data.get(f, 0) for f in radar_feats]
        fig_r = go.Figure(go.Scatterpolar(
            r=rvals + [rvals[0]],
            theta=[f.title() for f in radar_feats] + [radar_feats[0].title()],
            fill='toself', fillcolor='rgba(29,185,84,0.12)',
            line=dict(color='#1db954', width=2.5), marker=dict(color='#1db954', size=7)
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor='#0d0d10',
                radialaxis=dict(visible=True, range=[0,1], color='#333',
                                gridcolor='#1a1a24', tickfont=dict(color='#444')),
                angularaxis=dict(color='#888', gridcolor='#1a1a24')
            ),
            paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
            height=320, margin=dict(t=20,b=20,l=60,r=60)
        )
        st.plotly_chart(fig_r, use_container_width=True)

    st.divider()
    bcol, _ = st.columns([1, 2])
    with bcol:
        go_btn = st.button("🎵  Analyse Hit Potential")

    if go_btn:
        try:
            inp_df     = pd.DataFrame([input_data])
            inp_scaled = scaler.transform(inp_df)
            prob       = model.predict_proba(inp_scaled)[:, 1][0]
            pred       = 1 if prob >= threshold else 0

            st.divider()
            if pred == 1:
                st.markdown("""<div class="hit-banner">
                    <div class="banner-icon">🔥</div>
                    <div class="banner-title">THIS SONG IS A HIT!</div>
                    <div class="banner-sub">High chart potential detected</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="miss-banner">
                    <div class="banner-icon">📉</div>
                    <div class="banner-title">NOT LIKELY A HIT</div>
                    <div class="banner-sub">Low chart probability</div>
                </div>""", unsafe_allow_html=True)

            p1, p2, p3, p4 = st.columns(4)
            conf = abs(prob - threshold) / (1 - threshold) if prob >= threshold else abs(prob - threshold) / threshold
            with p1: st.markdown(f'<div class="card"><div class="card-value">{prob:.1%}</div><div class="card-label">Hit Probability</div></div>', unsafe_allow_html=True)
            with p2: st.markdown(f'<div class="card"><div class="card-value">{"HIT" if pred==1 else "MISS"}</div><div class="card-label">Verdict</div></div>', unsafe_allow_html=True)
            with p3: st.markdown(f'<div class="card"><div class="card-value">{threshold:.0%}</div><div class="card-label">Threshold</div></div>', unsafe_allow_html=True)
            with p4: st.markdown(f'<div class="card"><div class="card-value">{conf:.0%}</div><div class="card-label">Confidence</div></div>', unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number=dict(suffix="%", font=dict(color="white", size=42, family="Syne")),
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor="#333", tickfont=dict(color="#555")),
                    bar=dict(color="#1db954" if pred==1 else "#e8175d", thickness=0.65),
                    bgcolor="#0d0d10", borderwidth=0,
                    steps=[
                        dict(range=[0, threshold*100], color="#111114"),
                        dict(range=[threshold*100, 100], color="rgba(29,185,84,0.06)")
                    ],
                    threshold=dict(line=dict(color="white",width=2), thickness=0.75, value=threshold*100)
                ),
                title=dict(text="Hit Probability Score", font=dict(color="#888",size=16,family="DM Sans"))
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
                height=260, margin=dict(t=60,b=10,l=40,r=40)
            )
            st.plotly_chart(fig_g, use_container_width=True)

            if data_loaded and features:
                common = [f for f in features if f in df.columns]
                if common:
                    st.markdown('<div class="sec-head">📊 Your Song vs Dataset Average</div>', unsafe_allow_html=True)
                    avg_vals  = df[common].mean().values
                    your_vals = [input_data[f] for f in common]
                    fig_cmp = go.Figure()
                    fig_cmp.add_trace(go.Bar(name="Your Song", x=common, y=your_vals,
                                             marker_color="#1db954", opacity=0.9))
                    fig_cmp.add_trace(go.Bar(name="Dataset Avg", x=common, y=avg_vals,
                                             marker_color="#e8175d", opacity=0.7))
                    fig_cmp.update_layout(
                        barmode="group", template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d10",
                        font=dict(color="white"), legend=dict(bgcolor="rgba(0,0,0,0)"),
                        xaxis_tickangle=-30, height=340, margin=dict(t=20,b=60)
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ═══════════════════════════════════════════════════════════
#  📊 ANALYSIS
# ═══════════════════════════════════════════════════════════
elif menu == "📊  Analysis":
    st.title("Data Analysis")
    st.markdown("##### Explore every dimension of the Spotify dataset")
    st.divider()

    if not data_loaded:
        st.error("Dataset not found."); st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Distributions","🔥 Correlation","🎻 Feature vs Hit","🗺 Scatter Matrix","📦 Box Plots"])

    with tab1:
        col_sel = st.selectbox("Feature", NUM_COLS, key="d1")
        fig_h = px.histogram(
            df, x=col_sel,
            color=HIT_COL if (HIT_COL and col_sel != HIT_COL) else None,
            barmode="overlay", nbins=50, opacity=0.78,
            color_discrete_map={0:"#e8175d", 1:"#1db954"},
            template="plotly_dark"
        )
        fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d10",
                            font=dict(color="white"), height=360)
        st.plotly_chart(fig_h, use_container_width=True)
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Summary Statistics**")
            st.dataframe(df[NUM_COLS].describe().round(3), use_container_width=True)
        with cb:
            if "popularity" in df.columns:
                fig_pop = px.box(df, y="popularity",
                                 color=HIT_COL if HIT_COL else None,
                                 color_discrete_map={0:"#e8175d",1:"#1db954"},
                                 template="plotly_dark", title="Popularity Distribution")
                fig_pop.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d10",
                                      font=dict(color="white"))
                st.plotly_chart(fig_pop, use_container_width=True)

    with tab2:
        corr = df[NUM_COLS].dropna().corr()
        fig_corr = px.imshow(
            corr, color_continuous_scale=[[0,"#e8175d"],[0.5,"#111"],[1,"#1db954"]],
            zmin=-1, zmax=1, template="plotly_dark", title="Feature Correlation Matrix"
        )
        fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=580)
        st.plotly_chart(fig_corr, use_container_width=True)
        if HIT_COL and HIT_COL in corr.columns:
            hcorr = corr[HIT_COL].drop(HIT_COL).sort_values()
            fig_hc = px.bar(x=hcorr.values, y=hcorr.index, orientation="h",
                            color=hcorr.values,
                            color_continuous_scale=[[0,"#e8175d"],[0.5,"#333"],[1,"#1db954"]],
                            template="plotly_dark", title=f"Correlation with '{HIT_COL}'")
            fig_hc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d10",
                                 font=dict(color="white"), showlegend=False,
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_hc, use_container_width=True)

    with tab3:
        if HIT_COL:
            fc = st.selectbox("Feature", [c for c in NUM_COLS if c != HIT_COL], key="fvh")
            fig_v = px.violin(df, y=fc, x=HIT_COL, color=HIT_COL, box=True,
                              color_discrete_map={0:"#e8175d",1:"#1db954"},
                              template="plotly_dark", title=f"{fc.title()} · Hit vs Not Hit")
            fig_v.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d10",
                                font=dict(color="white"))
            st.plotly_chart(fig_v, use_container_width=True)
            mean_df = df.groupby(HIT_COL)[NUM_COLS].mean().T.reset_index()
            mean_df.columns = ["Feature","Not Hit","Hit"]
            melt = mean_df.melt(id_vars="Feature", var_name="Category", value_name="Mean")
            fig_m = px.bar(melt, x="Feature", y="Mean", color="Category", barmode="group",
                           color_discrete_map={"Hit":"#1db954","Not Hit":"#e8175d"},
                           template="plotly_dark", title="Mean Feature Values: Hit vs Not Hit")
            fig_m.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d10",
                                font=dict(color="white"), xaxis_tickangle=-30,
                                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("No hit/target column detected in dataset.")

    with tab4:
        sel = st.multiselect("Features", NUM_COLS,
                             default=NUM_COLS[:5] if len(NUM_COLS)>=5 else NUM_COLS)
        if len(sel) >= 2:
            samp = df.sample(min(1500,len(df)), random_state=42)
            fig_sm = px.scatter_matrix(samp, dimensions=sel, color=HIT_COL,
                                       color_discrete_map={0:"#e8175d",1:"#1db954"},
                                       template="plotly_dark", title="Scatter Matrix")
            fig_sm.update_traces(diagonal_visible=False, marker=dict(size=3,opacity=0.45))
            fig_sm.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                 font=dict(color="white"), height=700)
            st.plotly_chart(fig_sm, use_container_width=True)

    with tab5:
        bp_feat = st.selectbox("Feature", NUM_COLS, key="bp")
        if ARTIST_COL:
            top20 = (df.groupby(ARTIST_COL).size()
                     .sort_values(ascending=False).head(20).index.tolist())
            fig_bp = px.box(
                df[df[ARTIST_COL].isin(top20)], x=ARTIST_COL, y=bp_feat,
                color=HIT_COL if HIT_COL else None,
                color_discrete_map={0:"#e8175d",1:"#1db954"},
                template="plotly_dark", title=f"{bp_feat.title()} by Artist (Top 20)"
            )
            fig_bp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d10",
                                 font=dict(color="white"), xaxis_tickangle=-40, height=480)
            st.plotly_chart(fig_bp, use_container_width=True)
        else:
            fig_bp2 = px.box(df, y=bp_feat, color=HIT_COL if HIT_COL else None,
                             color_discrete_map={0:"#e8175d",1:"#1db954"},
                             template="plotly_dark")
            fig_bp2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d10",
                                  font=dict(color="white"))
            st.plotly_chart(fig_bp2, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  🎤 ARTIST INSIGHTS
# ═══════════════════════════════════════════════════════════
elif menu == "🎤  Artist Insights":
    st.title("Artist Insights")
    st.markdown("##### Deep-dive into artist performance & hit patterns")
    st.divider()

    if not data_loaded:
        st.error("Dataset not found."); st.stop()

    if not ARTIST_COL:
        st.warning("⚠️ No artist column found. Expected: `artist`, `artist_name`, `artists`, `performer`.")
        st.stop()

    total_artists = df[ARTIST_COL].nunique()
    if HIT_COL:
        hit_artists = df[df[HIT_COL]==1][ARTIST_COL].nunique()
        art_hits    = df[df[HIT_COL]==1].groupby(ARTIST_COL).size()
        top_artist  = art_hits.idxmax() if len(art_hits) > 0 else "N/A"
        top_hits    = art_hits.max()    if len(art_hits) > 0 else 0
    else:
        hit_artists = top_hits = 0; top_artist = "N/A"

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="card"><div class="card-value">{total_artists:,}</div><div class="card-label">Total Artists</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="card"><div class="card-value">{hit_artists:,}</div><div class="card-label">Hit Artists</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="card"><div class="card-value">{top_hits}</div><div class="card-label">Most Hits</div><div class="card-sub">{str(top_artist)[:18]}</div></div>', unsafe_allow_html=True)
    with c4:
        spa = len(df) / max(total_artists,1)
        st.markdown(f'<div class="card"><div class="card-value">{spa:.1f}</div><div class="card-label">Avg Songs/Artist</div></div>', unsafe_allow_html=True)

    st.divider()

    if HIT_COL:
        st.markdown('<div class="sec-head">🏆 Top 20 Artists by Hit Count</div>', unsafe_allow_html=True)
        top20h = (df[df[HIT_COL]==1].groupby(ARTIST_COL)
                  .size().reset_index(name="hits")
                  .sort_values("hits", ascending=False).head(20))
        fig_a1 = px.bar(top20h, x="hits", y=ARTIST_COL, orientation="h",
                        color="hits",
                        color_continuous_scale=[[0,"#0d5e29"],[1,"#1db954"]],
                        template="plotly_dark")
        fig_a1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d10",
                             font=dict(color="white"), showlegend=False,
                             yaxis=dict(automargin=True), height=520,
                             coloraxis_showscale=False)
        st.plotly_chart(fig_a1, use_container_width=True)

    st.markdown('<div class="sec-head">📈 Artist Hit Rate (min 5 songs)</div>', unsafe_allow_html=True)
    if HIT_COL:
        art_stats = df.groupby(ARTIST_COL).agg(
            total=(ARTIST_COL,"count"), hits=(HIT_COL,"sum")
        ).reset_index()
        art_stats = art_stats[art_stats["total"] >= 5].copy()
        art_stats["hit_rate"] = art_stats["hits"] / art_stats["total"] * 100
        art_stats = art_stats.sort_values("hit_rate", ascending=False).head(25)
        fig_a2 = px.bar(art_stats, x=ARTIST_COL, y="hit_rate",
                        color="hit_rate",
                        color_continuous_scale=[[0,"#e8175d"],[0.5,"#f5a623"],[1,"#1db954"]],
                        template="plotly_dark",
                        hover_data={"total":True,"hits":True})
        fig_a2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d10",
                             font=dict(color="white"), xaxis_tickangle=-35,
                             coloraxis_showscale=False, height=400)
        st.plotly_chart(fig_a2, use_container_width=True)

    # Audio fingerprint radar
    radar_cols = [c for c in ["danceability","energy","speechiness","acousticness",
                               "instrumentalness","liveness","valence"] if c in df.columns]
    if radar_cols:
        st.markdown('<div class="sec-head">🎛 Artist Audio Fingerprint Comparison</div>', unsafe_allow_html=True)
        top10_artists = (df.groupby(ARTIST_COL).size()
                         .sort_values(ascending=False).head(10).index.tolist())
        sel_artists = st.multiselect("Select Artists to Compare", top10_artists,
                                     default=top10_artists[:3])
        if sel_artists:
            COLORS = ["#1db954","#e8175d","#f5a623","#1da1f2","#9b59b6",
                      "#e67e22","#1abc9c","#e74c3c","#3498db","#2ecc71"]
            fig_af = go.Figure()
            for i, art in enumerate(sel_artists):
                adf  = df[df[ARTIST_COL] == art]
                vals = [adf[c].mean() for c in radar_cols]
                r, g, b = int(COLORS[i%len(COLORS)][1:3],16), int(COLORS[i%len(COLORS)][3:5],16), int(COLORS[i%len(COLORS)][5:7],16)
                fig_af.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=[c.title() for c in radar_cols] + [radar_cols[0].title()],
                    fill='toself', name=art,
                    line=dict(color=COLORS[i%len(COLORS)], width=2),
                    fillcolor=f"rgba({r},{g},{b},0.08)"
                ))
            fig_af.update_layout(
                polar=dict(
                    bgcolor="#0d0d10",
                    radialaxis=dict(visible=True, range=[0,1], color="#333",
                                    gridcolor="#1a1a24", tickfont=dict(color="#444")),
                    angularaxis=dict(color="#888", gridcolor="#1a1a24")
                ),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
                legend=dict(bgcolor="rgba(0,0,0,0)"), height=420,
                margin=dict(t=20,b=20,l=60,r=60)
            )
            st.plotly_chart(fig_af, use_container_width=True)

    # Artist search
    st.markdown('<div class="sec-head">🔍 Artist Deep Dive</div>', unsafe_allow_html=True)
    search = st.text_input("Search artist", placeholder="e.g. Drake, Taylor Swift…")
    if search:
        mask = df[ARTIST_COL].str.contains(search, case=False, na=False)
        adf  = df[mask]
        if len(adf) == 0:
            st.warning("No artist found.")
        else:
            cols_show = ([TRACK_COL] if TRACK_COL else []) + [ARTIST_COL] + \
                        ([HIT_COL] if HIT_COL else []) + \
                        [c for c in radar_cols if c in adf.columns]
            st.markdown(f"**{len(adf)} songs found**")
            st.dataframe(adf[cols_show].head(50), use_container_width=True)
            if HIT_COL:
                hr = adf[HIT_COL].mean() * 100
                st.markdown(f"""<div class="ibox">
                <b>Hit Rate:</b> {hr:.1f}% &nbsp;|&nbsp;
                <b>Total Songs:</b> {len(adf)} &nbsp;|&nbsp;
                <b>Hits:</b> {int(adf[HIT_COL].sum())}
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  🔬 MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════
elif menu == "🔬  Model Insights":
    st.title("Model Insights")
    st.markdown("##### Understand what drives the model's predictions")
    st.divider()

    if not model_loaded:
        st.error("Model not loaded."); st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="card"><div class="card-value">{threshold:.0%}</div><div class="card-label">Threshold</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="card"><div class="card-value">{len(features)}</div><div class="card-label">Features</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="card"><div class="card-value">LR</div><div class="card-label">Algorithm</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="card"><div class="card-value">SS</div><div class="card-label">Scaler</div></div>', unsafe_allow_html=True)

    st.divider()

    try:
        coefs = model.coef_[0]
        cdf = (pd.DataFrame({"Feature":features,"Coefficient":coefs})
               .assign(Abs=lambda x: x.Coefficient.abs())
               .sort_values("Coefficient"))
        fig_c = px.bar(cdf, x="Coefficient", y="Feature", orientation="h",
                       color="Coefficient",
                       color_continuous_scale=[[0,"#e8175d"],[0.5,"#333"],[1,"#1db954"]],
                       template="plotly_dark",
                       title="🧠 Feature Coefficients (positive → pushes toward Hit)")
        fig_c.add_vline(x=0, line_color="#555", line_dash="dot")
        fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d10",
                            font=dict(color="white"), showlegend=False,
                            coloraxis_showscale=False, height=460,
                            yaxis=dict(automargin=True))
        st.plotly_chart(fig_c, use_container_width=True)

        st.markdown('<div class="sec-head">Feature Impact Table</div>', unsafe_allow_html=True)
        disp = (cdf[["Feature","Coefficient","Abs"]]
                .sort_values("Abs",ascending=False).reset_index(drop=True))
        disp.columns = ["Feature","Coefficient","Absolute Impact"]
        st.dataframe(disp.style.background_gradient(cmap="RdYlGn", subset=["Coefficient"]),
                     use_container_width=True)
    except Exception:
        st.info("Coefficient view not available for this model type.")

    st.divider()
    st.markdown('<div class="sec-head">📐 Precision / Recall / F1 vs Threshold</div>', unsafe_allow_html=True)
    if data_loaded and HIT_COL:
        try:
            samp = df.dropna(subset=features+[HIT_COL]).sample(min(2000,len(df)), random_state=42)
            Xs   = scaler.transform(samp[features])
            ps   = model.predict_proba(Xs)[:,1]
            yt   = samp[HIT_COL].values
            ts   = np.linspace(0.1,0.9,60)
            prec, rec, f1s = [], [], []
            for t in ts:
                pr = (ps>=t).astype(int)
                tp = ((pr==1)&(yt==1)).sum(); fp = ((pr==1)&(yt==0)).sum(); fn = ((pr==0)&(yt==1)).sum()
                p  = tp/(tp+fp) if tp+fp>0 else 0
                r  = tp/(tp+fn) if tp+fn>0 else 0
                f  = 2*p*r/(p+r) if p+r>0 else 0
                prec.append(p); rec.append(r); f1s.append(f)
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=ts,y=prec,name="Precision",line=dict(color="#1db954",width=2.5)))
            fig_t.add_trace(go.Scatter(x=ts,y=rec, name="Recall",   line=dict(color="#e8175d",width=2.5)))
            fig_t.add_trace(go.Scatter(x=ts,y=f1s, name="F1 Score", line=dict(color="#f5a623",width=2.5,dash="dash")))
            fig_t.add_vline(x=threshold, line_color="white", line_dash="dot",
                            annotation_text=f"Current ({threshold:.2f})",
                            annotation_font_color="white")
            fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d10",
                                font=dict(color="white"), template="plotly_dark",
                                xaxis_title="Threshold", yaxis_title="Score",
                                legend=dict(bgcolor="rgba(0,0,0,0)"), height=360)
            st.plotly_chart(fig_t, use_container_width=True)
        except Exception as e:
            st.info(f"Threshold analysis skipped: {e}")


# ═══════════════════════════════════════════════════════════
#  ℹ️ ABOUT
# ═══════════════════════════════════════════════════════════
elif menu == "ℹ️  About":
    st.title("About This Project")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        for title, content in [
            ("🎵 Project","Final Year ML Project predicting Spotify chart hits using audio features and Logistic Regression."),
            ("🧠 Model","• Algorithm: Logistic Regression<br>• Scaler: Standard Scaler<br>• Features: 13+ audio signals<br>• Output: Hit probability 0–1"),
            ("🛠 Stack","• <b>UI:</b> Streamlit<br>• <b>ML:</b> Scikit-learn<br>• <b>Viz:</b> Plotly<br>• <b>Data:</b> Pandas · NumPy"),
        ]:
            st.markdown(f'<div class="ibox" style="margin-bottom:12px"><b>{title}</b><br><br>{content}</div>', unsafe_allow_html=True)

    with c2:
        for title, content in [
            ("🎛 Audio Features","• <b>Danceability</b> – Rhythm suitability<br>• <b>Energy</b> – Intensity & power<br>• <b>Valence</b> – Musical mood<br>• <b>Tempo</b> – BPM<br>• <b>Loudness</b> – dB level<br>• <b>Speechiness</b> – Spoken words<br>• <b>Acousticness</b> – Acoustic signal<br>• <b>Instrumentalness</b> – No vocals<br>• <b>Liveness</b> – Live audience"),
            ("📚 Academic","Developed for Final Year academic submission demonstrating supervised binary classification on music industry data."),
        ]:
            st.markdown(f'<div class="ibox" style="margin-bottom:12px"><b>{title}</b><br><br>{content}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="text-align:center;padding:24px 0">
        <div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;
                    background:linear-gradient(90deg,#1db954,#1ed760);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            🎵 Spotify Hit Predictor
        </div>
        <div style="color:#333;font-size:12px;margin-top:8px">
            Made with ❤️ · Final Year ML Project · Streamlit + Scikit-learn + Plotly
        </div>
    </div>
    """, unsafe_allow_html=True)
