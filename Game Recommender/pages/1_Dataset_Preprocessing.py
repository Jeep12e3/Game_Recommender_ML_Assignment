import pandas as pd
import streamlit as st

from src.data_loader import data_help_message, load_games
from src.preprocessing import prepare_games
from src.ui import metric_row, page_setup, require_data


page_setup("Dataset & Preprocessing")
st.title("Dataset & Preprocessing")

raw_df = load_games()
if not require_data(raw_df):
    st.info(data_help_message())
    st.stop()

df = prepare_games(raw_df)

st.subheader("Raw Dataset")
metric_row(
    [
        ("Rows", f"{len(raw_df):,}"),
        ("Columns", f"{raw_df.shape[1]:,}"),
        ("Duplicate Names", f"{raw_df['name'].duplicated().sum():,}"),
        ("Missing Descriptions", f"{raw_df['short_description'].isna().sum():,}"),
    ]
)

with st.expander("Raw data preview", expanded=False):
    st.dataframe(raw_df.head(30), use_container_width=True)

st.subheader("Preprocessing Steps")
st.markdown(
    """
    - Removed duplicate `appid` and duplicate game names.
    - Converted `release_date` into `release_year`.
    - Created `is_free` from the price column.
    - Parsed `genres`, `categories`, `developers`, `publishers`, and weighted Steam `tags`.
    - Created `total_reviews` and `rating_percent`.
    - Converted estimated owner ranges into numeric midpoint values.
    - Applied `MinMaxScaler` to numeric fields used in the recommendation score.
    """
)

st.subheader("Prepared Dataset")
metric_row(
    [
        ("Rows After Cleaning", f"{len(df):,}"),
        ("Release Range", f"{df['release_year'].min()} - {df['release_year'].max()}"),
        ("Average Rating", f"{df['rating_percent'].mean():.1f}%"),
        ("Average Price", f"${df['price'].mean():.2f}"),
    ]
)

preview_columns = [
    "appid",
    "name",
    "release_year",
    "price",
    "is_free",
    "genres_list",
    "tags_list",
    "rating_percent",
    "total_reviews",
    "owner_midpoint",
]
st.dataframe(df[preview_columns].head(50), use_container_width=True)

st.subheader("Scaled Numeric Features")
scaled_columns = [
    "rating_percent",
    "rating_percent_scaled",
    "total_reviews",
    "total_reviews_scaled",
    "owner_midpoint",
    "owner_midpoint_scaled",
    "release_year",
    "release_year_scaled",
]
st.dataframe(df[scaled_columns].sample(min(20, len(df)), random_state=42), use_container_width=True)

st.caption(
    "MinMaxScaler converts each numeric feature into a 0-1 range, so large values like review counts "
    "do not overpower smaller values like rating percentage."
)

