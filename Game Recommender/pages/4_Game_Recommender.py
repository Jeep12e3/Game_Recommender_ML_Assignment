import streamlit as st

from src.config import DEFAULT_FEATURES, DEFAULT_WEIGHTS
from src.data_loader import active_preprocessing_key, load_prepared_games
from src.preprocessing import apply_filters
from src.recommender import build_vector_model, recommend_games
from src.ui import game_card, page_setup, require_data


page_setup("Game Recommender")
st.title("Game Recommender")

df = load_prepared_games()
if not require_data(df):
    st.stop()
data_key = active_preprocessing_key()

selected_features = st.session_state.get("selected_features", DEFAULT_FEATURES)
score_weights = st.session_state.get("score_weights", DEFAULT_WEIGHTS)
feature_key = tuple(sorted(selected_features.items()))
_, matrix = build_vector_model(df, feature_key, data_key)

st.sidebar.header("Recommendation Filters")
all_genres = sorted({genre for genres in df["genres_list"] for genre in genres})
platform = st.sidebar.selectbox("Platform", ["Any", "windows", "mac", "linux"])
genre = st.sidebar.selectbox("Genre", ["Any"] + all_genres)
price_type = st.sidebar.selectbox("Price Type", ["Any", "Free", "Paid"])
min_year = int(df["release_year"].min())
max_year = int(df["release_year"].max())
year_range = st.sidebar.slider("Release Year", min_year, max_year, (min_year, max_year))
min_rating = st.sidebar.slider("Minimum Positive Rating", 0, 100, 0)
min_reviews = st.sidebar.number_input("Minimum Review Count", min_value=0, value=100, step=100)
top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

candidate_df = apply_filters(
    df,
    platform=platform,
    genre=genre,
    price_type=price_type,
    year_range=year_range,
    min_rating=min_rating,
    min_reviews=min_reviews,
)

st.write(f"Candidate games after filters: **{len(candidate_df):,}**")

game_name = st.selectbox(
    "Choose a game you like",
    df.sort_values("name")["name"].tolist(),
    index=None,
    placeholder="Search for a Steam game...",
)

if not game_name:
    st.info("Select a game to generate recommendations.")
    st.stop()

if st.button("Recommend Games", type="primary"):
    recommendations = recommend_games(
        df=df,
        candidate_df=candidate_df,
        selected_game=game_name,
        matrix=matrix,
        weights=score_weights,
        top_n=top_n,
    )

    if recommendations.empty:
        st.warning("No recommendations found. Try relaxing the filters.")
        st.stop()

    st.subheader(f"Because you liked {game_name}")
    for _, row in recommendations.iterrows():
        game_card(row)
