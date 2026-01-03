🎧 Spotify Tracks Feature Analysis & Visualization
📌 Project Overview

This project explores audio characteristics of Spotify tracks to understand how different musical features vary across genres and influence listening patterns. Using a cleaned Spotify dataset and an interactive Tableau dashboard, the analysis highlights genre-level trends, feature distributions, and comparative insights that help explain what makes music sound energetic, danceable, or calm.

The final output is an interactive Tableau dashboard that enables users to visually compare audio features such as energy, danceability, valence, tempo, and loudness across genres.

🔗 Interactive Dashboard:
👉 https://public.tableau.com/app/profile/mamta.jha/viz/Final230/SpotifyTracksFeature

🔗 Dataset Source:
👉 https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

🎯 Objectives

Analyze Spotify audio features across thousands of tracks

Compare genre-wise musical characteristics

Identify patterns in energy, danceability, mood, and tempo

Build an interactive, user-friendly visualization for exploratory analysis

Translate raw music data into clear, interpretable insights

📂 Dataset Description

The dataset is sourced from Kaggle and contains detailed metadata for Spotify tracks.

Key Attributes:

track_name – Name of the song

artist_name – Artist or band

track_genre – Genre classification

popularity – Spotify popularity score (0–100)

danceability – Suitability for dancing (0–1)

energy – Intensity and activity level (0–1)

valence – Positivity or happiness of the track (0–1)

tempo – Beats per minute (BPM)

loudness – Overall loudness (dB)

acousticness, instrumentalness, speechiness – Audio composition metrics

🧹 Data Cleaning & Preprocessing

Before visualization, the dataset underwent thorough preprocessing to ensure accuracy and consistency:

Removed duplicate and incomplete records

Standardized text fields (track names, genres)

Verified valid ranges for all audio features (0–1 scale)

Filtered out extreme outliers in tempo and loudness

Ensured genre labels were consistent and analysis-ready

These steps ensured the dashboard reflects clean, reliable, and meaningful patterns.

📊 Visualization & Dashboard Design

The analysis is presented through an interactive Tableau dashboard built using Tableau.

Dashboard Highlights:

Genre-wise comparison of audio features

Visual patterns in energy, danceability, and valence

Interactive filters for genre exploration

Clear, intuitive layout for both technical and non-technical users

The dashboard allows users to quickly answer questions such as:

Which genres are the most energetic?

How does danceability vary across music styles?

Which genres tend to sound happier or calmer?

🔍 Key Insights

Electronic and Dance genres show consistently high energy and tempo

Acoustic and Folk genres score higher on acousticness and lower on loudness

Pop and Hip-Hop tracks balance energy and danceability, contributing to higher popularity

Valence varies significantly by genre, reflecting different emotional tones in music

These insights demonstrate how audio features shape the identity and listener perception of genres.

🛠 Tools & Technologies

Python – Data cleaning and preprocessing

Pandas & NumPy – Data manipulation

Tableau Public – Interactive visualization and dashboard creation

Kaggle Dataset – Data sourcing

📈 Applications & Use Cases

Music recommendation systems

Genre classification and trend analysis

Artist and label strategy insights

Academic data visualization projects

Portfolio demonstration of analytics + storytelling skills

🚀 Future Enhancements

Incorporate time-based analysis (release year trends)

Add popularity vs. feature correlation analysis

Apply clustering or PCA to group similar genres

Extend dashboard with user-driven comparisons
