import streamlit as st

from src.data_loader import data_help_message, load_games
from src.preprocessing import prepare_games
from src.ui import metric_row, page_setup


page_setup("Home")

st.title("Steam Game Recommender")
st.write(
    "A content-based recommender system that suggests Steam games using genres, tags, "
    "categories, descriptions, and optional popularity/rating signals."
)

raw_df = load_games()

if raw_df.empty:
    st.warning("Dataset not found.")
    st.info(data_help_message())
    st.stop()

df = prepare_games(raw_df)

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
        2. **EDA / Insights** explores trends, genres, prices, platforms, and reviews.
        3. **Build Recommender** lets users choose recommendation features and weights.
        4. **Game Recommender** returns similar games with match scores and explanations.
        """
    )

with right:
    st.subheader("Recommendation Method")
    st.write(
        "The app uses TF-IDF to represent game metadata as vectors, then compares games "
        "with cosine similarity. Numeric signals such as rating, popularity, and recency "
        "are normalized with MinMaxScaler and blended into the final match score."
    )

st.subheader("Dataset Preview")
st.dataframe(
    df[
        [
            "appid",
            "name",
            "release_date",
            "price",
            "genres",
            "tags",
            "rating_percent",
            "total_reviews",
        ]
    ].head(20),
    use_container_width=True,
)

