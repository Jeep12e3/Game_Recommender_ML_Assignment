from __future__ import annotations

import pandas as pd
import streamlit as st


PREVIEW_OPTIONS = [20, 50, 100, 300, 500, 1000, "All"]


def page_setup(title: str):
    st.set_page_config(
        page_title=f"{title} | Steam Game Recommender",
        page_icon="🎮",
        layout="wide",
    )


def require_data(df: pd.DataFrame) -> bool:
    if not df.empty:
        return True
    st.warning("Dataset not found yet.")
    st.markdown(
        "Place `games_march2025_cleaned.csv` in the `data/` folder, "
        "or place the Kaggle zip as `data/archive (2).zip`."
    )
    return False


def metric_row(metrics: list[tuple[str, str]]):
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)


def preview_dataframe(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    label: str = "Rows to preview",
    key: str | None = None,
):
    selected = st.selectbox(label, PREVIEW_OPTIONS, index=1, key=key)
    display_df = df[columns] if columns else df

    if selected == "All":
        st.warning(
            "Showing all rows can slow down the web page. Downloading the full data is usually smoother."
        )
        preview_df = display_df
    else:
        preview_df = display_df.head(int(selected))

    st.dataframe(preview_df, use_container_width=True)


@st.cache_data(show_spinner=False)
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def download_dataframe(df: pd.DataFrame, file_name: str, label: str):
    st.download_button(
        label=label,
        data=convert_df_to_csv(df),
        file_name=file_name,
        mime="text/csv",
    )


def game_card(row: pd.Series):
    with st.container(border=True):
        cols = st.columns([1.1, 2.4])
        image = row.get("header_image", "")
        if isinstance(image, str) and image.startswith("http"):
            cols[0].image(image, use_container_width=True)
        else:
            cols[0].empty()

        price = "Free" if bool(row.get("is_free", False)) else f"${row.get('price', 0):.2f}"
        cols[1].subheader(row.get("name", "Unknown game"))
        cols[1].caption(
            f"{price} | {int(row.get('release_year', 0))} | "
            f"{row.get('rating_percent', 0):.0f}% positive | "
            f"{int(row.get('total_reviews', 0)):,} reviews"
        )
        if "match_score" in row:
            cols[1].progress(float(row["match_score"]) / 100, text=f"Match Score: {row['match_score']:.0f}%")
        if row.get("shared_reasons"):
            cols[1].write(f"Recommended because: {row['shared_reasons']}")
        description = str(row.get("short_description", "")).strip()
        if description:
            cols[1].write(description[:280] + ("..." if len(description) > 280 else ""))
        appid = row.get("appid")
        if pd.notna(appid):
            cols[1].link_button("Open on Steam", f"https://store.steampowered.com/app/{int(appid)}")
