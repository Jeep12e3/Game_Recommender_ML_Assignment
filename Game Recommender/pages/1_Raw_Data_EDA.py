import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import data_help_message, load_games
from src.preprocessing import parse_list_text, parse_tags
from src.ui import download_dataframe, metric_row, page_setup, preview_dataframe, require_data


page_setup("Raw Data & EDA")
st.title("Raw Data & EDA")

raw_df = load_games()
if not require_data(raw_df):
    st.info(data_help_message())
    st.stop()

raw_df = raw_df.copy()
raw_df["release_date"] = pd.to_datetime(raw_df["release_date"], errors="coerce")
raw_df["release_year"] = raw_df["release_date"].dt.year
raw_df["price"] = pd.to_numeric(raw_df["price"], errors="coerce").fillna(0)
raw_df["total_reviews_raw"] = (
    pd.to_numeric(raw_df.get("positive", 0), errors="coerce").fillna(0)
    + pd.to_numeric(raw_df.get("negative", 0), errors="coerce").fillna(0)
)

metric_row(
    [
        ("Raw Rows", f"{len(raw_df):,}"),
        ("Columns", f"{raw_df.shape[1]:,}"),
        ("Duplicate Names", f"{raw_df['name'].duplicated().sum():,}"),
        ("Missing Descriptions", f"{raw_df['short_description'].isna().sum():,}"),
    ]
)

st.subheader("Raw Data Preview")
preview_columns = [
    "appid",
    "name",
    "release_date",
    "price",
    "genres",
    "tags",
    "positive",
    "negative",
    "estimated_owners",
]
preview_dataframe(raw_df, columns=preview_columns, key="raw_preview_rows")
download_dataframe(raw_df, "raw_steam_games.csv", "Download full raw data")

st.subheader("Raw Data Quality")
left, right = st.columns(2)
with left:
    missing = raw_df.isna().sum().sort_values(ascending=False).head(15).reset_index()
    missing.columns = ["column", "missing_values"]
    st.dataframe(missing, use_container_width=True)
with right:
    dtypes = raw_df.dtypes.astype(str).reset_index()
    dtypes.columns = ["column", "data_type"]
    st.dataframe(dtypes, use_container_width=True)

tab_release, tab_genres, tab_reviews, tab_platforms = st.tabs(
    ["Release Trends", "Genres & Tags", "Reviews & Popularity", "Platforms & Price"]
)

with tab_release:
    release_counts = (
        raw_df[raw_df["release_year"].between(1997, 2025)]
        .groupby("release_year")
        .size()
        .reset_index(name="games")
    )
    st.plotly_chart(
        px.line(release_counts, x="release_year", y="games", markers=True, title="Raw Games Released Per Year"),
        use_container_width=True,
    )

with tab_genres:
    genre_counts = raw_df["genres"].apply(parse_list_text).explode().dropna().value_counts().head(20).reset_index()
    genre_counts.columns = ["genre", "games"]
    st.plotly_chart(
        px.bar(genre_counts, x="games", y="genre", orientation="h", title="Top Raw Genres"),
        use_container_width=True,
    )

    tag_counts = raw_df["tags"].apply(lambda value: parse_tags(value, limit=10)).explode().dropna().value_counts().head(20).reset_index()
    tag_counts.columns = ["tag", "games"]
    st.plotly_chart(
        px.bar(tag_counts, x="games", y="tag", orientation="h", title="Top Raw Steam Tags"),
        use_container_width=True,
    )

with tab_reviews:
    reviewed = raw_df[raw_df["total_reviews_raw"].gt(0)].copy()
    reviewed["rating_percent_raw"] = (
        pd.to_numeric(reviewed["positive"], errors="coerce").fillna(0)
        / reviewed["total_reviews_raw"]
        * 100
    )
    st.plotly_chart(
        px.histogram(reviewed, x="rating_percent_raw", nbins=40, title="Raw Review Score Distribution"),
        use_container_width=True,
    )
    top_reviewed = raw_df.sort_values("total_reviews_raw", ascending=False).head(15)
    st.plotly_chart(
        px.bar(top_reviewed, x="total_reviews_raw", y="name", orientation="h", title="Most Reviewed Raw Games"),
        use_container_width=True,
    )

with tab_platforms:
    platform_counts = {
        "Windows": int(raw_df["windows"].sum()),
        "Mac": int(raw_df["mac"].sum()),
        "Linux": int(raw_df["linux"].sum()),
    }
    st.plotly_chart(
        px.bar(x=list(platform_counts.keys()), y=list(platform_counts.values()), title="Raw Platform Support"),
        use_container_width=True,
    )

    price_counts = raw_df["price"].eq(0).map({True: "Free", False: "Paid"}).value_counts().reset_index()
    price_counts.columns = ["type", "games"]
    st.plotly_chart(px.pie(price_counts, names="type", values="games", title="Raw Free vs Paid Games"), use_container_width=True)

