# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_ensemble import ensemble_scores

st.set_page_config(page_title="Board Game Recommender", layout="wide")
st.title("Board Game Recommender")
st.caption("Hybrid recommendations powered by Collaborative Filtering (CF), Content-Based Filtering (CBF), and LLM models.")

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_csv("../data/games.csv")

@st.cache_data
def load_mechanics():
    mechanics_df = pd.read_csv(
        "../data/game_mechanics.csv", header=None, names=["mechanic"]
    )
    return mechanics_df["mechanic"].dropna().sort_values().tolist()

@st.cache_data
def load_categories():
    categories_df = pd.read_csv(
        "../data/game_categories.csv", header=None, names=["category"]
    )
    return categories_df["category"].dropna().sort_values().tolist()

@st.cache_data
def load_game_types():
    game_types_df = pd.read_csv(
        "../data/game_types.csv", header=None, names=["type"]
    )
    return game_types_df["type"].dropna().sort_values().tolist()

games_df = load_data()
mechanics_options = load_mechanics()
categories_options = load_categories()
game_type_options = load_game_types()

# ========== SIDEBAR ==========
st.sidebar.header("Your Preferences")

# --- CF inputs ---
liked_games = st.sidebar.multiselect("Liked Board Games", games_df["Name"].unique())
disliked_games = st.sidebar.multiselect("Disliked Board Games", games_df["Name"].unique())

# --- Filter inputs ---
year_range = st.sidebar.slider("Year Published", 1990, 2021, (2000, 2021))
rating_min = st.sidebar.slider("Minimum Rating", 1.0, 10.0, 6.0)

# --- CBF inputs ---
min_players, max_players = st.sidebar.slider(
    "Player Count Range", 1, 20, (2, 4)
)
play_time = st.sidebar.selectbox("Play Time", ["<30 mins", "30–60 mins", "60–90 mins", "90–120 mins", ">120 mins"])
complexity = st.sidebar.slider("Complexity", 1.0, 5.0, (2.3, 3.6), 0.1)
mechanics = st.sidebar.multiselect("Game Mechanics", mechanics_options)
categories = st.sidebar.multiselect("Game Category", categories_options)
game_type = st.sidebar.multiselect("Game Type", game_type_options)

# --- LLM input ---
description = st.sidebar.text_area("Describe the kind of board game you enjoy", placeholder="Example: I like strategic games with some luck and engine building mechanics.")

# ========== FILTER DISPLAY ==========
filtered_df = games_df.copy()

if "YearPublished" in filtered_df.columns:
    filtered_df = filtered_df.query("YearPublished >= @year_range[0] and YearPublished <= @year_range[1]")
if "AvgRating" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["AvgRating"] >= rating_min]
if {"MinPlayers", "MaxPlayers"} <= set(filtered_df.columns):
    filtered_df = filtered_df.query("MinPlayers <= @max_players and MaxPlayers >= @min_players")

st.subheader("Games Matching Your Basic Filters")
st.write(f"{len(filtered_df)} games found.")
if not filtered_df.empty:
    st.dataframe(filtered_df[["Name", "YearPublished", "AvgRating"]].head(10))
    if "Category" in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(7, 3))
        filtered_df["Category"].value_counts().head(10).sort_values().plot(kind="barh", ax=ax, color="#2E86DE")
        ax.set_title("Top Game Categories")
        ax.set_xlabel("Count")
        st.pyplot(fig)

# ========== HYBRID RECOMMENDATIONS ==========
st.markdown("---")
st.subheader("AI-Powered Hybrid Recommendations")

if st.button("Get Recommendations"):
    with st.spinner("Running ensemble model..."):
        user_prefs = {
            "liked_games": liked_games,
            "disliked_games": disliked_games,
            "category": categories,
            "mechanics": mechanics,
            "game_type": game_type,
            "min_players": min_players,
            "max_players": max_players,
            "play_time": play_time,
            "complexity": complexity,
        }

        try:
            rec_df = ensemble_scores(
                user_id=1,
                user_description=description,
                user_preferences=user_prefs,
                top_n=10,
            )
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
        else:
            if rec_df.empty:
                st.warning("No recommendations found for these preferences.")
            else:
                st.dataframe(
                    rec_df[["Name", "Composite_Score", "CF_Score", "CBF_Score", "LLM_Score"]],
                    use_container_width=True,
                )
