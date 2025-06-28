# UAS-CC
🎧 Music Recommender App
A simple web-based application built with Streamlit to recommend songs based on user mood or similarity to another song, using Spotify track features and machine learning techniques like KMeans clustering and cosine similarity.

🔧 Features
🎵 Mood-Based Recommendation
Select a mood (e.g., Relax, Sad, Party) and get top 5 songs that match the mood.

🔁 Song Similarity Recommender
Enter a song title, and the app finds 5 most similar tracks using cosine similarity.

📊 KMeans Clustering
Each track is automatically assigned to one of 6 audio-based clusters.

☁️ Cloud-Ready Dataset Loader
Large dataset (>25MB) is hosted on Google Drive and downloaded on the fly using gdown.
