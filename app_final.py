import streamlit as st
import pandas as pd
import os
import gdown
import base64
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# === Spotify API Auth ===
def get_spotify_token(client_id, client_secret):
    auth_url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}"
    }
    data = {
        "grant_type": "client_credentials"
    }
    response = requests.post(auth_url, headers=headers, data=data)
    return response.json().get('access_token', None)

def get_track_info(track_id, token):
    url = f"https://api.spotify.com/v1/tracks/{track_id}"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return None
    data = r.json()
    return {
        "name": data['name'],
        "artist": data['artists'][0]['name'],
        "album": data['album']['name'],
        "image": data['album']['images'][0]['url'],
        "spotify_url": data['external_urls']['spotify'],
        "preview": data.get('preview_url', None)
    }

@st.cache_data
def download_data():
    url = "https://drive.google.com/uc?id=1Ysov4hAioJY7BvKXYd11GQlPpJPVoSR2"
    out = "SpotifyFeatures.csv"
    if not os.path.exists(out):
        gdown.download(url, out, quiet=False)
    return pd.read_csv(out)

@st.cache_data
def load_data():
    df = download_data()
    df = df.dropna().drop_duplicates()
    drop_cols = ['duration_ms','instrumentalness','key','liveness','loudness','speechiness','time_signature']
    df = df.drop(columns=drop_cols)
    return df

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

# === Load and Prepare Dataset ===
df = load_data()
df['mood'] = df.apply(get_mood, axis=1)
features = df[['valence','energy','danceability','tempo','acousticness']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)
df['cluster'] = KMeans(n_clusters=6, random_state=42).fit_predict(scaled)

if 'start_idx' not in st.session_state:
    st.session_state.start_idx = 0
if 'prev_mood' not in st.session_state:
    st.session_state.prev_mood = ""

client_id = st.secrets["SPOTIFY_CLIENT_ID"]
client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
token = get_spotify_token(client_id, client_secret)

# === UI Header ===
st.set_page_config(page_title="Moodtify", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #1DB954;'>🎧 Moodtify</h1>
    <p style='text-align: center;'>Karena Musik Tak Pernah Salah Menyuarakan Perasaan</p>
    <hr>
""", unsafe_allow_html=True)

# === Mode Selection on Main Screen ===
mode = st.radio("Pilih Mode:", ["💫 Jelajahi Mood-mu", "📻 Rasa yang Sama, Lagu Berbeda"], horizontal=True)

if mode == "💫 Jelajahi Mood-mu":
    mood = st.selectbox("Pilih Moodmu saat ini:", sorted(df['mood'].unique()))
    genre = st.selectbox("Pilih Genremu:", ["(Semua)"] + sorted(df['genre'].unique()))

    if mood != st.session_state.prev_mood:
        st.session_state.start_idx = 0
        st.session_state.prev_mood = mood

    result = df[df['mood'] == mood]
    if genre != "(Semua)":
        result = result[result['genre'].str.lower() == genre.lower()]
    result = result.drop_duplicates(subset=['track_name','artist_name']).sort_values(by='popularity',ascending=False)

    page_size = 5
    start = st.session_state.start_idx
    end = start + page_size
    paginated = result.iloc[start:end]

    for _, row in paginated.iterrows():
        st.markdown(f"### {row['track_name']} - {row['artist_name']}")
        if token and pd.notna(row['track_id']):
            track_data = get_track_info(row['track_id'], token)
            if track_data:
                st.image(track_data['image'], width=250)
                st.markdown(f"[🔗 Dengarkan di Spotify]({track_data['spotify_url']})")
                if track_data['preview']:
                    st.audio(track_data['preview'])
        st.markdown("---")

    if end < len(result):
        if st.button("Lihat 5 Rekomendasi Berikutnya"):
            st.session_state.start_idx += page_size

elif mode == "📻 Rasa yang Sama, Lagu Berbeda":
    input_song = st.text_input("Masukkan Judul Lagu Favoritmu:")
    if input_song:
        match = df[df['track_name'].str.lower() == input_song.lower()]
        if not match.empty:
            target = match.iloc[0]
            sim = cosine_similarity([scaled[match.index[0]]], scaled)[0]
            df['similarity'] = sim
            result = df[df['track_name'].str.lower() != input_song.lower()].sort_values(by='similarity', ascending=False).head(5)

            for _, row in result.iterrows():
                st.markdown(f"### {row['track_name']} - {row['artist_name']}")
                if token and pd.notna(row['track_id']):
                    track_data = get_track_info(row['track_id'], token)
                    if track_data:
                        st.image(track_data['image'], width=250)
                        st.markdown(f"[🔗 Dengarkan di Spotify]({track_data['spotify_url']})")
                        if track_data['preview']:
                            st.audio(track_data['preview'])
                st.markdown("---")
        else:
            st.warning("Lagu tidak ditemukan dalam database.")
