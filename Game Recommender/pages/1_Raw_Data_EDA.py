import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import data_help_message, load_games
from src.preprocessing import parse_list_text, parse_tags
from src.ui import download_dataframe, metric_row, page_setup, preview_dataframe, require_data


LIST_LIKE_COLUMNS = {
    "genres",
    "tags",
    "categories",
    "developers",
    "publishers",
}


page_setup("Raw Data & EDA")
st.title("Raw Data & EDA")
st.caption(
    "This page starts from the raw loaded dataset. The recommender itself uses the cleaned/prepared "
    "dataset shown on Home and configured on the Preprocessing page."
)

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
st.caption("Use the column toggle to switch between the readable preview subset and the full raw table.")
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
preview_dataframe(raw_df, columns=preview_columns, key="raw_preview")
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


def show_column_explorer(df: pd.DataFrame) -> None:
    st.subheader("Any Column Explorer")
    st.caption("Pick any raw column to inspect missingness, sample values, distributions, and top values.")

    controls = st.columns([1.6, 1, 1])
    selected_column = controls[0].selectbox("Column", df.columns.tolist(), index=df.columns.get_loc("name") if "name" in df else 0)
    top_n = controls[1].slider("Top values", 5, 50, 20, 5)
    bins = controls[2].slider("Histogram bins", 10, 80, 35, 5)

    series = df[selected_column]
    missing_count = int(series.isna().sum())
    missing_percent = missing_count / max(len(series), 1) * 100
    unique_count = int(series.nunique(dropna=True))

    metric_row(
        [
            ("Data Type", str(series.dtype)),
            ("Missing", f"{missing_count:,} ({missing_percent:.1f}%)"),
            ("Unique Values", f"{unique_count:,}"),
            ("Rows", f"{len(series):,}"),
        ]
    )

    st.markdown("**Sample Values**")
    sample_values = series.dropna().astype(str).head(12).reset_index(drop=True).to_frame(name=selected_column)
    st.dataframe(sample_values, use_container_width=True)

    if selected_column in LIST_LIKE_COLUMNS:
        parser = parse_tags if selected_column == "tags" else parse_list_text
        exploded = series.apply(parser).explode().dropna()
        exploded = exploded[exploded.astype(str).str.strip().ne("")]
        value_counts = exploded.value_counts().head(top_n).reset_index()
        value_counts.columns = [selected_column, "count"]
        st.markdown("**Parsed Top Values**")
        st.dataframe(value_counts, use_container_width=True)
        if not value_counts.empty:
            st.plotly_chart(
                px.bar(value_counts, x="count", y=selected_column, orientation="h", title=f"Top {selected_column}"),
                use_container_width=True,
            )
        return

    if pd.api.types.is_datetime64_any_dtype(series):
        dates = series.dropna()
        if dates.empty:
            st.info("This date column has no valid values to chart.")
            return
        date_summary = pd.DataFrame(
            {
                "metric": ["earliest", "latest"],
                "value": [dates.min(), dates.max()],
            }
        )
        st.dataframe(date_summary, use_container_width=True)
        by_year = dates.dt.year.value_counts().sort_index().reset_index()
        by_year.columns = ["year", "rows"]
        st.plotly_chart(px.line(by_year, x="year", y="rows", markers=True, title=f"{selected_column} by Year"), use_container_width=True)
        return

    numeric_series = pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_numeric_dtype(series) or numeric_series.notna().sum() >= max(20, len(series) * 0.5):
        numeric_series = numeric_series.dropna()
        if numeric_series.empty:
            st.info("This numeric column has no valid values to chart.")
            return
        st.markdown("**Summary Statistics**")
        st.dataframe(numeric_series.describe().to_frame(name=selected_column), use_container_width=True)
        chart_df = numeric_series.to_frame(name=selected_column)
        st.plotly_chart(px.histogram(chart_df, x=selected_column, nbins=bins, title=f"{selected_column} Distribution"), use_container_width=True)
        st.plotly_chart(px.box(chart_df, x=selected_column, title=f"{selected_column} Spread"), use_container_width=True)
        return

    text_series = series.dropna().astype(str).str.strip()
    text_series = text_series[text_series.ne("")]
    value_counts = text_series.value_counts().head(top_n).reset_index()
    value_counts.columns = [selected_column, "count"]
    st.markdown("**Top Values**")
    st.dataframe(value_counts, use_container_width=True)
    if not value_counts.empty:
        st.plotly_chart(
            px.bar(value_counts, x="count", y=selected_column, orientation="h", title=f"Top {selected_column} Values"),
            use_container_width=True,
        )


tab_release, tab_genres, tab_reviews, tab_platforms, tab_explorer = st.tabs(
    ["Release Trends", "Genres & Tags", "Reviews & Popularity", "Platforms & Price", "Any Column Explorer"]
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

with tab_explorer:
    show_column_explorer(raw_df)
