from __future__ import annotations

import pandas as pd
import streamlit as st


PREVIEW_OPTIONS = [20, 50, 100, 300, 500, 1000, "All"]


APP_CSS = """
<style>
:root {
    --steam-ink: #17212f;
    --steam-green: #1b7f5a;
    --steam-cyan: #2f9fd8;
    --steam-amber: #e7a928;
    --surface: #ffffff;
    --soft-surface: #eef4f2;
}

.stApp {
    background:
        linear-gradient(180deg, rgba(27, 127, 90, 0.12) 0, rgba(247, 249, 248, 0) 240px),
        #f5f8f6;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: var(--steam-ink) !important;
}

h1 {
    font-weight: 800;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff 0%, #eef7f4 100%);
    border: 1px solid rgba(27, 127, 90, 0.18);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    box-shadow: 0 8px 22px rgba(23, 33, 47, 0.06);
}


.stApp, .stApp p, .stApp li, .stApp span, .stApp label,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMetricLabel"],
[data-testid="stCaptionContainer"],
[data-testid="stSidebar"] * {
    color: var(--steam-ink) !important;
}

[data-testid="stSidebar"] [aria-selected="true"] {
    background: var(--steam-green) !important;
    border-radius: 8px;
}

[data-testid="stSidebar"] [aria-selected="true"],
[data-testid="stSidebar"] [aria-selected="true"] * {
    color: #ffffff !important;
}

[data-testid="stMetricValue"] div,
[data-testid="stMetricDelta"] div {
    color: var(--steam-green) !important;
}

.stButton > button[kind="primary"],
.stButton > button[kind="primary"] *,
.stDownloadButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] * {
    color: #ffffff !important;
}
[data-testid="stMetricValue"] {
    color: var(--steam-green);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
}

.stTabs [data-baseweb="tab"] {
    background: #ffffff;
    border: 1px solid rgba(23, 33, 47, 0.10);
    border-radius: 8px 8px 0 0;
    padding-left: 1rem;
    padding-right: 1rem;
}

.stTabs [aria-selected="true"] {
    border-top: 3px solid var(--steam-green);
}

.stDataFrame, [data-testid="stTable"] {
    border: 1px solid rgba(23, 33, 47, 0.08);
    border-radius: 8px;
    overflow: hidden;
}

[data-testid="stSidebar"] {
    background: #edf4f1;
}

.stButton > button, .stDownloadButton > button, .stLinkButton > a {
    border-radius: 8px;
    border-color: rgba(27, 127, 90, 0.35);
}
</style>
"""


def page_setup(title: str):
    st.set_page_config(
        page_title=f"{title} | Steam Game Recommender",
        page_icon="🎮",
        layout="wide",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)


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
    allow_all_columns: bool = True,
):
    row_key = f"{key}_rows" if key else None
    column_key = f"{key}_columns" if key else None
    selected = st.selectbox(label, PREVIEW_OPTIONS, index=1, key=row_key)

    preview_columns = [column for column in (columns or df.columns.tolist()) if column in df.columns]
    if allow_all_columns and columns:
        column_mode = st.radio(
            "Columns to show",
            ["Selected preview columns", "All columns"],
            horizontal=True,
            key=column_key,
        )
        display_df = df if column_mode == "All columns" else df[preview_columns]
    else:
        display_df = df[preview_columns] if columns else df

    if selected == "All":
        st.warning(
            "Showing all rows can slow down the web page. Downloading the full data is usually smoother."
        )
        preview_df = display_df
    else:
        preview_df = display_df.head(int(selected))

    st.dataframe(preview_df, use_container_width=True)
    st.caption(f"Showing {preview_df.shape[0]:,} rows and {preview_df.shape[1]:,} of {df.shape[1]:,} columns.")


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
        if row.get("score_breakdown"):
            cols[1].caption(f"Score mix: {row['score_breakdown']}")
        description = str(row.get("short_description", "")).strip()
        if description:
            cols[1].write(description[:280] + ("..." if len(description) > 280 else ""))
        appid = row.get("appid")
        if pd.notna(appid):
            cols[1].link_button("Open on Steam", f"https://store.steampowered.com/app/{int(appid)}")


