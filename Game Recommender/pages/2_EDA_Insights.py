import plotly.express as px
import streamlit as st

from src.data_loader import load_games
from src.preprocessing import prepare_games
from src.ui import metric_row, page_setup, require_data


page_setup("EDA / Insights")
st.title("EDA / Insights")

raw_df = load_games()
if not require_data(raw_df):
    st.stop()

df = prepare_games(raw_df)

metric_row(
    [
        ("Games", f"{len(df):,}"),
        ("Median Price", f"${df['price'].median():.2f}"),
        ("Median Rating", f"{df['rating_percent'].median():.0f}%"),
        ("Windows Games", f"{df['windows'].sum():,}"),
    ]
)

tab_release, tab_genres, tab_reviews, tab_platforms = st.tabs(
    ["Release Trends", "Genres & Tags", "Reviews & Popularity", "Platforms & Price"]
)

with tab_release:
    release_counts = (
        df[df["release_year"].between(1997, 2025)]
        .groupby("release_year")
        .size()
        .reset_index(name="games")
    )
    st.plotly_chart(
        px.line(release_counts, x="release_year", y="games", markers=True, title="Games Released Per Year"),
        use_container_width=True,
    )

with tab_genres:
    genre_counts = df["genres_list"].explode().dropna().value_counts().head(20).reset_index()
    genre_counts.columns = ["genre", "games"]
    st.plotly_chart(
        px.bar(genre_counts, x="games", y="genre", orientation="h", title="Top Genres"),
        use_container_width=True,
    )

    tag_counts = df["tags_list"].explode().dropna().value_counts().head(20).reset_index()
    tag_counts.columns = ["tag", "games"]
    st.plotly_chart(
        px.bar(tag_counts, x="games", y="tag", orientation="h", title="Top Steam Tags"),
        use_container_width=True,
    )

with tab_reviews:
    reviewed = df[df["total_reviews"].gt(0)]
    st.plotly_chart(
        px.histogram(reviewed, x="rating_percent", nbins=40, title="Review Score Distribution"),
        use_container_width=True,
    )
    top_reviewed = df.sort_values("total_reviews", ascending=False).head(15)
    st.plotly_chart(
        px.bar(top_reviewed, x="total_reviews", y="name", orientation="h", title="Most Reviewed Games"),
        use_container_width=True,
    )

with tab_platforms:
    platform_counts = {
        "Windows": int(df["windows"].sum()),
        "Mac": int(df["mac"].sum()),
        "Linux": int(df["linux"].sum()),
    }
    st.plotly_chart(
        px.bar(x=list(platform_counts.keys()), y=list(platform_counts.values()), title="Platform Support"),
        use_container_width=True,
    )

    price_counts = df["is_free"].map({True: "Free", False: "Paid"}).value_counts().reset_index()
    price_counts.columns = ["type", "games"]
    st.plotly_chart(px.pie(price_counts, names="type", values="games", title="Free vs Paid Games"), use_container_width=True)

