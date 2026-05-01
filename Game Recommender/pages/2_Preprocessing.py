import pandas as pd
import streamlit as st

from src.config import PROCESSED_PARQUET
from src.data_loader import (
    clear_prepared_cache,
    data_help_message,
    load_base_clean_games,
    load_games,
    load_prepared_games,
    save_active_preprocessing_options,
)
from src.preprocessing import (
    default_preprocessing_options,
    normalize_preprocessing_options,
)
from src.ui import download_dataframe, metric_row, page_setup, preview_dataframe, require_data


page_setup("Preprocessing")
st.title("Preprocessing")

raw_df = load_games()
if not require_data(raw_df):
    st.info(data_help_message())
    st.stop()

raw_years = pd.to_datetime(raw_df["release_date"], errors="coerce").dt.year.dropna()
min_year = int(raw_years.min())
max_year = int(raw_years.max())

current_options = normalize_preprocessing_options(
    st.session_state.get("preprocessing_options"),
    raw_df,
)

st.write(
    "Choose how the dataset should be cleaned before building the recommender. "
    "Adult/sexual-content games are always removed by default."
)

with st.form("preprocessing_form"):
    st.subheader("Preprocessing Options")

    left, middle, right = st.columns(3)
    remove_duplicates = left.checkbox(
        "Remove duplicate games",
        value=bool(current_options["remove_duplicates"]),
        help="Removes duplicate app IDs and duplicate game names.",
    )
    missing_descriptions = middle.selectbox(
        "Missing descriptions",
        ["Keep as empty", "Remove missing descriptions"],
        index=["Keep as empty", "Remove missing descriptions"].index(current_options["missing_descriptions"]),
    )
    tag_limit = right.selectbox(
        "Steam tags used per game",
        [5, 10, 20, 30, 50],
        index=[5, 10, 20, 30, 50].index(int(current_options["tag_limit"])),
        help="Fewer tags are cleaner; more tags give the recommender more detail.",
    )

    left, middle, right = st.columns(3)
    min_reviews = left.number_input(
        "Minimum review count",
        min_value=0,
        value=int(current_options["min_reviews"]),
        step=100,
    )
    min_rating = middle.slider(
        "Minimum positive rating",
        min_value=0,
        max_value=100,
        value=int(current_options["min_rating"]),
    )
    price_type = right.selectbox(
        "Price type",
        ["All", "Free only", "Paid only"],
        index=["All", "Free only", "Paid only"].index(current_options["price_type"]),
    )

    year_range = st.slider(
        "Release year range",
        min_year,
        max_year,
        tuple(current_options["year_range"] or (min_year, max_year)),
    )

    platforms = st.multiselect(
        "Keep games available on these platforms",
        ["windows", "mac", "linux"],
        default=list(current_options["platforms"]),
        help="A game is kept if it supports at least one selected platform.",
    )

    apply_clicked = st.form_submit_button("Apply Preprocessing", type="primary")
    reset_clicked = st.form_submit_button("Reset to Defaults")

if reset_clicked:
    save_active_preprocessing_options(default_preprocessing_options(raw_df))
    st.success("Preprocessing options reset to defaults. Reloading with default settings.")
    st.rerun()

if apply_clicked:
    if not platforms:
        st.error("Select at least one platform.")
        st.stop()

    save_active_preprocessing_options(
        {
            "remove_duplicates": remove_duplicates,
            "missing_descriptions": missing_descriptions,
            "min_reviews": min_reviews,
            "min_rating": min_rating,
            "year_range": year_range,
            "platforms": tuple(platforms),
            "price_type": price_type,
            "tag_limit": tag_limit,
        }
    )
    st.success("Preprocessing options applied. Build/Recommender pages will use this processed dataset.")
    st.rerun()

base_df = load_base_clean_games()
df = load_prepared_games()
active_settings = normalize_preprocessing_options(st.session_state.get("preprocessing_options"), raw_df)
mature_removed = len(raw_df) - len(base_df)
duplicates_removed = len(raw_df) - len(raw_df.drop_duplicates(subset=["appid"]).drop_duplicates(subset=["name"]))
total_removed = len(raw_df) - len(df)

st.subheader("Scaled Numeric Features")
scaled_columns = [
    "rating_percent",
    "rating_percent_scaled",
    "total_reviews",
    "total_reviews_scaled",
    "owner_midpoint",
    "owner_midpoint_scaled",
    "peak_ccu",
    "peak_ccu_scaled",
    "release_year",
    "release_year_scaled",
]
preview_dataframe(df, columns=scaled_columns, label="Scaled rows to preview", key="scaled_preview_rows")

st.caption(
    "MinMaxScaler converts each numeric feature into a 0-1 range, so large values like review counts "
    "do not overpower smaller values like rating percentage."
)

st.subheader("Current Preprocessing Summary")
metric_row(
    [
        ("Raw Rows", f"{len(raw_df):,}"),
        ("Prepared Rows", f"{len(df):,}"),
        ("Rows Removed", f"{total_removed:,}"),
        ("Mature Content Removed", f"{mature_removed:,}"),
    ]
)

with st.expander("Active preprocessing settings", expanded=False):
    st.json(active_settings)
st.caption(f"Duplicate rows detected in raw data: {duplicates_removed:,}.")

st.subheader("Preprocessed Data Preview")
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
preview_dataframe(df, columns=preview_columns, key="preprocessed_preview_rows")
download_dataframe(df, "preprocessed_steam_games.csv", "Download full preprocessed data")

st.subheader("Preprocessing Steps")
st.markdown(
    """
    - Removed adult/sexual-content games by default.
    - Converted `release_date` into `release_year`.
    - Created `is_free` from the price column.
    - Parsed `genres`, `categories`, `developers`, `publishers`, and weighted Steam `tags`.
    - Created `total_reviews` and `rating_percent`.
    - Converted estimated owner ranges into numeric midpoint values.
    - Applied the user-selected duplicate, missing-description, review, rating, year, platform, price, and tag-limit options.
    - Applied MinMaxScaler to numeric fields used in the recommendation score.
    """
)

if PROCESSED_PARQUET.exists():
    st.info(
        f"Default processed cache exists: `{PROCESSED_PARQUET.name}`. "
        "Custom preprocessing choices are cached in memory during the Streamlit session."
    )

if st.button("Clear Prepared Data Cache", type="primary"):
    clear_prepared_cache()
    st.success("Prepared cache cleared. Reload the page to rebuild it.")
