import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import data_help_message, load_games
from src.preprocessing import count_mature_content_games, parse_list_text, parse_tags
from src.ui import download_dataframe, metric_row, page_setup, preview_dataframe, require_data


LIST_LIKE_COLUMNS = {
    "genres",
    "tags",
    "categories",
    "developers",
    "publishers",
}


page_setup("Exploratory Data Analysis")
st.title("Exploratory Data Analysis")
st.caption(
    "This page explores the original Steam dataset before preprocessing. The goal is to understand "
    "data quality, feature completeness, review coverage, and distribution patterns that will affect "
    "the preprocessing choices and the recommender model."
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

st.subheader("Data Preview")
st.caption("The full raw table is shown by default. Switch to selected preview columns for a cleaner view.")
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
preview_dataframe(raw_df, columns=preview_columns, key="raw_preview", default_all_columns=True)
download_dataframe(raw_df, "raw_steam_games.csv", "Download full raw data")

left, right = st.columns(2)
with left:
    st.subheader("Data Missing Values")
    missing = raw_df.isna().sum().sort_values(ascending=False).reset_index()
    missing.columns = ["column", "missing_values"]
    missing["missing_percent"] = missing["missing_values"] / max(len(raw_df), 1) * 100
    st.dataframe(missing, width="stretch")
with right:
    st.subheader("Data Type")
    dtypes = raw_df.dtypes.astype(str).reset_index()
    dtypes.columns = ["column", "data_type"]
    st.dataframe(dtypes, width="stretch")


@st.cache_data(show_spinner=False)
def mature_content_estimate(df: pd.DataFrame) -> int:
    return count_mature_content_games(df)


def blank_or_missing(series: pd.Series) -> pd.Series:
    stripped = series.fillna("").astype(str).str.strip()
    return series.isna() | stripped.eq("") | stripped.eq("[]") | stripped.eq("{}")


def parsed_count(series: pd.Series, parser) -> pd.Series:
    return series.apply(lambda value: len(parser(value)))


def show_model_readiness(df: pd.DataFrame) -> None:
    st.subheader("Recommender Feature Readiness")
    st.caption(
        "This section highlights how complete the main recommender features are before preprocessing. "
        "It helps us decide which columns are reliable enough to use for content similarity."
    )

    readiness_rows = []
    feature_columns = [
        ("genres", parse_list_text, "Main content feature"),
        ("tags", lambda value: parse_tags(value, limit=50), "Main content feature"),
        ("categories", parse_list_text, "Main content feature"),
        ("short_description", None, "Main text feature"),
        ("developers", parse_list_text, "Optional content feature"),
        ("publishers", parse_list_text, "Optional content feature"),
    ]
    for column, parser, role in feature_columns:
        if column not in df:
            continue
        empty_mask = parsed_count(df[column], parser).eq(0) if parser else blank_or_missing(df[column])
        available = len(df) - int(empty_mask.sum())
        readiness_rows.append(
            {
                "feature": column,
                "role": role,
                "available_rows": available,
                "missing_or_empty": int(empty_mask.sum()),
                "available_percent": available / max(len(df), 1) * 100,
            }
        )

    st.dataframe(pd.DataFrame(readiness_rows), width="stretch")


def show_column_explorer(df: pd.DataFrame) -> None:
    st.subheader("Column Explorer")
    st.caption(
        "Pick any raw column to inspect missingness, sample values, distributions, and top values. Numeric columns "
        "use all valid rows for statistics and charts; text/list columns show samples and top values because plotting "
        "tens of thousands of categories would be hard to read."
    )

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
    st.dataframe(sample_values, width="stretch")

    if selected_column in LIST_LIKE_COLUMNS:
        parser = parse_tags if selected_column == "tags" else parse_list_text
        exploded = series.apply(parser).explode().dropna()
        exploded = exploded[exploded.astype(str).str.strip().ne("")]
        value_counts = exploded.value_counts().head(top_n).reset_index()
        value_counts.columns = [selected_column, "count"]
        st.markdown("**Parsed Top Values**")
        st.dataframe(value_counts, width="stretch")
        if not value_counts.empty:
            st.plotly_chart(
                px.bar(value_counts, x="count", y=selected_column, orientation="h", title=f"Top {selected_column}"),
                width="stretch",
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
        st.dataframe(date_summary, width="stretch")
        by_year = dates.dt.year.value_counts().sort_index().reset_index()
        by_year.columns = ["year", "rows"]
        st.plotly_chart(px.line(by_year, x="year", y="rows", markers=True, title=f"{selected_column} by Year"), width="stretch")
        return

    numeric_series = pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_numeric_dtype(series) or numeric_series.notna().sum() >= max(20, len(series) * 0.5):
        numeric_series = numeric_series.dropna()
        if numeric_series.empty:
            st.info("This numeric column has no valid values to chart.")
            return
        st.markdown("**Summary Statistics**")
        st.dataframe(numeric_series.describe().to_frame(name=selected_column), width="stretch")
        chart_df = numeric_series.to_frame(name=selected_column)
        st.plotly_chart(px.histogram(chart_df, x=selected_column, nbins=bins, title=f"{selected_column} Distribution"), width="stretch")
        st.plotly_chart(px.box(chart_df, x=selected_column, title=f"{selected_column} Spread"), width="stretch")
        return

    text_series = series.dropna().astype(str).str.strip()
    text_series = text_series[text_series.ne("")]
    value_counts = text_series.value_counts().head(top_n).reset_index()
    value_counts.columns = [selected_column, "count"]
    st.markdown("**Top Values**")
    st.dataframe(value_counts, width="stretch")
    if not value_counts.empty:
        st.plotly_chart(
            px.bar(value_counts, x="count", y=selected_column, orientation="h", title=f"Top {selected_column} Values"),
            width="stretch",
        )


show_model_readiness(raw_df)

st.subheader("Data Visualization")
tab_explorer, tab_release, tab_genres, tab_reviews, tab_platforms = st.tabs(
    [
        "Column Explorer",
        "Release Trends",
        "Genres & Tags",
        "Reviews & Popularity",
        "Platforms & Price",
    ]
)

with tab_explorer:
    show_column_explorer(raw_df)

with tab_release:
    release_counts = (
        raw_df[raw_df["release_year"].between(1997, 2025)]
        .groupby("release_year")
        .size()
        .reset_index(name="games")
    )
    st.plotly_chart(
        px.line(release_counts, x="release_year", y="games", markers=True, title="Data Games Released Per Year"),
        width="stretch",
    )

with tab_genres:
    mature_estimate = mature_content_estimate(raw_df)
    st.metric("Potential Mature Games", f"{mature_estimate:,}")

    genre_counts = raw_df["genres"].apply(parse_list_text).explode().dropna().value_counts().head(20).reset_index()
    genre_counts.columns = ["genre", "games"]
    st.plotly_chart(
        px.bar(genre_counts, x="games", y="genre", orientation="h", title="Top Data Genres"),
        width="stretch",
    )

    tag_counts = raw_df["tags"].apply(lambda value: parse_tags(value, limit=10)).explode().dropna().value_counts().head(20).reset_index()
    tag_counts.columns = ["tag", "games"]
    st.plotly_chart(
        px.bar(tag_counts, x="games", y="tag", orientation="h", title="Top Data Steam Tags"),
        width="stretch",
    )

with tab_reviews:
    review_coverage = pd.DataFrame(
        {
            "review_status": ["Has reviews", "No reviews"],
            "games": [int(raw_df["total_reviews_raw"].gt(0).sum()), int(raw_df["total_reviews_raw"].eq(0).sum())],
        }
    )
    st.plotly_chart(
        px.bar(review_coverage, x="review_status", y="games", title="Games With vs Without Reviews"),
        width="stretch",
    )

    reviewed = raw_df[raw_df["total_reviews_raw"].gt(0)].copy()
    reviewed["rating_percent_raw"] = (
        pd.to_numeric(reviewed["positive"], errors="coerce").fillna(0)
        / reviewed["total_reviews_raw"]
        * 100
    )
    st.plotly_chart(
        px.histogram(reviewed, x="rating_percent_raw", nbins=40, title="Data Review Score Distribution"),
        width="stretch",
    )
    top_reviewed = raw_df.sort_values("total_reviews_raw", ascending=False).head(15)
    st.plotly_chart(
        px.bar(top_reviewed, x="total_reviews_raw", y="name", orientation="h", title="Most Reviewed Data Games"),
        width="stretch",
    )

with tab_platforms:
    platform_counts = {
        "Windows": int(raw_df["windows"].sum()),
        "Mac": int(raw_df["mac"].sum()),
        "Linux": int(raw_df["linux"].sum()),
    }
    st.plotly_chart(
        px.bar(x=list(platform_counts.keys()), y=list(platform_counts.values()), title="Data Platform Support"),
        width="stretch",
    )

    price_counts = raw_df["price"].eq(0).map({True: "Free", False: "Paid"}).value_counts().reset_index()
    price_counts.columns = ["type", "games"]
    st.plotly_chart(px.pie(price_counts, names="type", values="games", title="Data Free vs Paid Games"), width="stretch")

