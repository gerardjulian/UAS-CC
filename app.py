import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
import gdown

# === Download Dataset dari Google Drive ===
@st.cache_data
def download_data():
    csv_file = "SpotifyFeatures.csv"
    if not os.path.exists(csv_file):
        file_id = "1Ysov4hAioJY7BvKXYd11GQlPpJPVoSR2"  # Ganti dengan ID file Google Drive
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, csv_file, quiet=False)
    return pd.read_csv(csv_file)

# === Load dan Bersihkan Dataset ===
@st.cache_data
def load_data():
    df = download_data()
    df = df.drop_duplicates().dropna()
    drop_cols = ['track_id', 'duration_ms', 'instrumentalness', 'key',
                 'liveness', 'loudness', 'speechiness', 'time_signature']
    df = df.drop(columns=drop_cols)
    return df

df = load_data()

# === Klasifikasi Mood ===
def get_mood(row):
    v, e, d, a, t = row['valence'], row['energy'], row['danceability'], row['acousticness'], row['tempo']
    if v > 0.65 and e > 0.70:
        return 'Happy'
    elif v < 0.30 and e < 0.45:
        return 'Sad'
    elif e > 0.75 and d > 0.65:
        return 'Party'
    elif e < 0.50 and a > 0.60 and v < 0.55:
        return 'Relax'
    elif e < 0.55 and v > 0.45 and t < 120:
        return 'Romantic'
    elif v < 0.40 and e < 0.60 and t < 110:
        return 'Melancholy'
    elif v > 0.50 and e > 0.55 and 100 <= t <= 130:
        return 'Inspired'
    else:
        return 'Neutral'

df['mood'] = df.apply(get_mood, axis=1)

# === Clustering dengan KMeans ===
features = df[['valence', 'energy', 'danceability', 'tempo', 'acousticness']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)
df['cluster'] = KMeans(n_clusters=6, random_state=42).fit_predict(scaled)

# === Sidebar Mode ===
mode = st.sidebar.selectbox("Mode Aplikasi", ["🎵 Mood-based Recommender", "🔁 Similar Song Recommender"])

if mode == "🎵 Mood-based Recommender":
    st.title("🎵 Rekomendasi Lagu Berdasarkan Mood")
    mood = st.selectbox("Pilih Mood:", sorted(df['mood'].unique()))
    genre = st.selectbox("Pilih Genre (Opsional):", ["(Semua)"] + sorted(df['genre'].unique()))
    
    result = df[df['mood'] == mood]
    if genre != "(Semua)":
        result = result[result['genre'].str.lower() == genre.lower()]
    result = result.drop_duplicates(subset=['track_name', 'artist_name'])
    result = result.sort_values(by='popularity', ascending=False).head(5)

    st.subheader(f"Top Lagu untuk Mood: {mood}")
    st.dataframe(result[['track_name', 'artist_name', 'genre', 'popularity']])

elif mode == "🔁 Similar Song Recommender":
    st.title("🔁 Rekomendasi Lagu Serupa")
    song_input = st.text_input("Masukkan Judul Lagu:")
    if song_input:
        selected = df[df['track_name'].str.lower() == song_input.lower()]
        if selected.empty:
            st.warning("Lagu tidak ditemukan.")
        else:
            query = scaler.transform(selected[features.columns])
            all_vecs = scaler.transform(df[features.columns])
            df['similarity'] = cosine_similarity(query, all_vecs)[0]
            result = df[df['track_name'].str.lower() != song_input.lower()]
            result = result.sort_values(by='similarity', ascending=False).head(5)

            st.subheader(f"Lagu Mirip dengan '{song_input.title()}'")
            st.dataframe(result[['track_name', 'artist_name', 'genre', 'popularity', 'similarity']])
