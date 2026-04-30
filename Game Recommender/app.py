import streamlit as st

from src.data_loader import data_help_message, load_prepared_games
from src.ui import metric_row, page_setup, preview_dataframe


page_setup("Home")

st.title("Steam Game Recommender")
st.write(
    "A content-based recommender system that suggests Steam games using genres, tags, "
    "categories, descriptions, and optional popularity/rating signals."
)

df = load_prepared_games()

if df.empty:
    st.warning("Dataset not found.")
    st.info(data_help_message())
    st.stop()

metric_row(
    [
        ("Games", f"{len(df):,}"),
        ("Release Years", f"{df['release_year'].min()} - {df['release_year'].max()}"),
        ("Free Games", f"{df['is_free'].sum():,}"),
        ("Paid Games", f"{(~df['is_free']).sum():,}"),
    ]
)

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Project Workflow")
    st.markdown(
        """
        1. **Dataset & Preprocessing** cleans the Steam dataset and prepares useful fields.
        2. **EDA / Insights** explores trends, genres, prices, platforms, reviews, and any selected column.
        3. **Build Recommender** lets users choose recommendation features and weights.
        4. **Game Recommender** returns similar games with match scores and explanations.
        """
    )

with right:
    st.subheader("Recommendation Method")
    st.write(
        "The app uses TF-IDF and shared genre/tag/category overlap to compare games. "
        "Rating, popularity, and recency signals are normalized and blended into the final score, "
        "with popularity dampened so famous games do not overpower closer matches."
    )

st.subheader("Prepared Dataset Preview")
st.caption(
    "This preview uses the cleaned/prepared dataset that powers recommendations. "
    "Switch to all columns when you want to inspect the full prepared table."
)
preview_dataframe(
    df,
    columns=[
        "appid",
        "name",
        "release_date",
        "price",
        "genres",
        "tags",
        "rating_percent",
        "total_reviews",
    ],
    key="home_preview",
)
