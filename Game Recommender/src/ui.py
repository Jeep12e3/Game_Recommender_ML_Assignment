from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import streamlit as st


PREVIEW_OPTIONS = [20, 50, 100, 300, 500, 1000, "All"]


def app_css() -> str:
    theme = {
        "ink": "#17212f",
        "muted": "#65727c",
        "green": "#1b7f5a",
        "cyan": "#2f9fd8",
        "amber": "#e7a928",
        "surface": "#ffffff",
        "soft_surface": "#eef4f2",
        "background": "#f5f8f6",
        "border": "rgba(23, 33, 47, 0.10)",
        "shadow": "rgba(23, 33, 47, 0.08)",
        "hero_a": "rgba(27, 127, 90, 0.18)",
        "hero_b": "rgba(47, 159, 216, 0.16)",
    }
    css = """
<style>
:root {
    --steam-ink: __INK__;
    --steam-muted: __MUTED__;
    --steam-green: __GREEN__;
    --steam-cyan: __CYAN__;
    --steam-amber: __AMBER__;
    --surface: __SURFACE__;
    --soft-surface: __SOFT_SURFACE__;
    --steam-bg: __BACKGROUND__;
    --steam-border: __BORDER__;
    --steam-shadow: __SHADOW__;
}

.stApp {
    background:
        radial-gradient(circle at 12% 0%, __HERO_A__ 0, transparent 330px),
        radial-gradient(circle at 88% 4%, __HERO_B__ 0, transparent 280px),
        var(--steam-bg);
}

[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stToolbar"] {
    background: var(--steam-bg) !important;
}

[data-testid="stHeader"]::before,
[data-testid="stHeader"]::after {
    background: transparent !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"],
[data-testid="stSpinner"] *,
.stSpinner,
.stSpinner * {
    color: var(--steam-ink) !important;
}

[data-testid="stSpinner"] > div,
.stSpinner > div {
    background: var(--surface) !important;
    border: 1px solid var(--steam-border) !important;
    border-radius: 8px !important;
    padding: 1rem 1.5rem !important;
    box-shadow: 0 8px 24px var(--steam-shadow) !important;
}

[data-testid="stSpinner"] p,
.stSpinner p {
    color: var(--steam-ink) !important;
    font-weight: 600 !important;
}

[data-testid="stSpinner"] svg,
[data-testid="stSpinner"] svg *,
[data-testid="stSpinner"] svg circle {
    stroke: var(--steam-green) !important;
    color: var(--steam-green) !important;
}

[data-testid="stToolbar"] button,
[data-testid="stToolbar"] button * {
    color: var(--steam-ink) !important;
}

/* ── Secondary / ghost buttons ── */
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
    background: var(--surface) !important;
    color: var(--steam-ink) !important;
    border: 1.5px solid rgba(47, 191, 131, 0.45) !important;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {
    background: var(--soft-surface) !important;
    border-color: var(--steam-green) !important;
    color: var(--steam-ink) !important;
}

.stButton > button[kind="secondary"] *,
.stButton > button:not([kind="primary"]) * {
    color: var(--steam-ink) !important;
}

/* ── Input fields (text, number, selectbox, multiselect) ── */
[data-baseweb="input"],
[data-baseweb="textarea"],
[data-baseweb="select"] > div,
[data-baseweb="select"] [data-testid="stSelectbox"],
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: var(--surface) !important;
    color: var(--steam-ink) !important;
    border-color: var(--steam-border) !important;
}

[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] input {
    color: var(--steam-ink) !important;
    background: transparent !important;
}

[data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"],
[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    background: var(--surface) !important;
    border: 1px solid var(--steam-border) !important;
}

[data-baseweb="menu"] li,
[data-baseweb="menu"] li * {
    color: var(--steam-ink) !important;
}

[data-baseweb="menu"] li:hover,
[data-baseweb="option"]:hover {
    background: var(--soft-surface) !important;
}

[data-baseweb="option"][aria-selected="true"] {
    background: rgba(47, 191, 131, 0.18) !important;
}

/* ── Checkbox & Radio ── */
.stCheckbox label,
.stCheckbox label span,
.stRadio label,
.stRadio label span {
    color: var(--steam-ink) !important;
}

.stCheckbox [data-testid="stWidgetLabel"],
.stRadio [data-testid="stWidgetLabel"] {
    color: var(--steam-ink) !important;
}

/* ── Slider ── */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
.stSlider [data-testid="stSliderThumbValue"] {
    color: var(--steam-muted) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--steam-border) !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] summary {
    color: var(--steam-ink) !important;
}

[data-testid="stExpander"] summary:hover {
    background: var(--soft-surface) !important;
    border-radius: 8px;
}

[data-testid="stExpander"] svg {
    fill: var(--steam-ink) !important;
}

/* ── Alert / Info / Warning / Success ── */
[data-testid="stAlert"],
[data-testid="stNotification"],
.stAlert {
    background: var(--surface) !important;
    border-radius: 8px !important;
    border: 1px solid var(--steam-border) !important;
}

[data-testid="stAlert"] *,
.stAlert * {
    color: var(--steam-ink) !important;
}

/* ── Code block ── */
[data-testid="stCode"],
[data-testid="stCode"] pre,
.stCode pre {
    background: var(--soft-surface) !important;
    color: var(--steam-ink) !important;
    border: 1px solid var(--steam-border) !important;
    border-radius: 8px !important;
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: var(--steam-ink) !important;
}

h1 {
    font-weight: 800;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--surface) 0%, var(--soft-surface) 100%);
    border: 1px solid var(--steam-border);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    box-shadow: 0 8px 22px var(--steam-shadow);
}


.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp small,
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
.stButton button[data-testid="baseButton-primary"],
.stButton button[data-testid="baseButton-primary"] *,
.stDownloadButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] *,
[data-testid="stDownloadButton"] button[data-testid="baseButton-primary"],
[data-testid="stDownloadButton"] button[data-testid="baseButton-primary"] *,
[data-testid="stFormSubmitButton"] > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] *,
[data-testid="stFormSubmitButton"] button,
[data-testid="stFormSubmitButton"] button *,
.stLinkButton > a,
.stLinkButton > a *,
[data-testid="stPageLink"] a,
[data-testid="stPageLink"] a * {
    color: #ffffff !important;
}
[data-testid="stMetricValue"] {
    color: var(--steam-green);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
}

.stTabs [data-baseweb="tab"] {
    background: var(--surface);
    border: 1px solid var(--steam-border);
    border-radius: 8px 8px 0 0;
    padding-left: 1rem;
    padding-right: 1rem;
}

.stTabs [aria-selected="true"] {
    border-top: 3px solid var(--steam-green);
}

.stDataFrame, [data-testid="stTable"] {
    border: 1px solid var(--steam-border);
    border-radius: 8px;
    overflow: hidden;
}

[data-testid="stSidebar"] {
    background: var(--soft-surface);
}

.stButton > button, .stDownloadButton > button, .stLinkButton > a {
    border-radius: 8px;
    border-color: rgba(27, 127, 90, 0.35);
}

.stButton > button[kind="primary"],
.stButton button[data-testid="baseButton-primary"],
.stDownloadButton > button[kind="primary"],
[data-testid="stDownloadButton"] button[data-testid="baseButton-primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"],
[data-testid="stFormSubmitButton"] button,
.stLinkButton > a,
[data-testid="stPageLink"] a {
    background: linear-gradient(135deg, var(--steam-green), #146849) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    box-shadow: 0 10px 22px rgba(27, 127, 90, 0.22);
    font-weight: 700;
    text-decoration: none !important;
}

.stButton > button[kind="primary"]:hover,
.stButton button[data-testid="baseButton-primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover,
[data-testid="stDownloadButton"] button[data-testid="baseButton-primary"]:hover,
[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] button:hover,
.stLinkButton > a:hover,
[data-testid="stPageLink"] a:hover {
    filter: brightness(1.06);
    border-color: rgba(255, 255, 255, 0.28) !important;
}

[data-testid="stPageLink"] {
    display: inline-flex;
    margin: 0.25rem 0 1rem;
}

[data-testid="stPageLink"] a {
    border-radius: 8px;
    padding: 0.55rem 0.9rem;
}

[data-baseweb="tag"] {
    background: var(--steam-green) !important;
    border-radius: 8px !important;
}

[data-baseweb="tag"] *,
[data-baseweb="tag"] svg,
[data-baseweb="tag"] svg * {
    color: #ffffff !important;
    fill: #ffffff !important;
}

.home-banner {
    min-height: 230px;
    margin-bottom: 1.5rem;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--steam-border);
    box-shadow: 0 18px 42px var(--steam-shadow);
    background:
        linear-gradient(90deg, rgba(13, 23, 21, 0.90), rgba(13, 23, 21, 0.54), rgba(13, 23, 21, 0.16)),
        repeating-linear-gradient(135deg, rgba(255, 255, 255, 0.10) 0 1px, transparent 1px 18px),
        radial-gradient(circle at 76% 34%, rgba(47, 159, 216, 0.45), transparent 130px),
        linear-gradient(135deg, #0f2a24 0%, #1b7f5a 46%, #203a64 100%);
    display: flex;
    align-items: end;
}

.home-banner.has-image {
    min-height: 310px;
    background-size: cover;
    background-position: center;
}

.home-banner__content {
    padding: 2rem;
    max-width: 760px;
}

.home-banner__eyebrow {
    color: #8ee4bd !important;
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.home-banner__title {
    color: #ffffff !important;
    font-size: clamp(2rem, 5vw, 4.4rem);
    line-height: 0.96;
    font-weight: 900;
    margin: 0;
}

.home-banner__copy {
    color: rgba(255, 255, 255, 0.82) !important;
    max-width: 560px;
    margin-top: 0.9rem;
    font-size: 1rem;
}

.game-card {
    display: grid;
    grid-template-columns: minmax(240px, 34%) 1fr;
    gap: 1.25rem;
    background: var(--surface);
    border: 1px solid var(--steam-border);
    border-radius: 8px;
    box-shadow: 0 12px 28px var(--steam-shadow);
    padding: 1rem;
    margin: 0.75rem 0 1rem;
}

.game-card__media {
    min-height: 260px;
    border-radius: 8px;
    overflow: hidden;
    background: linear-gradient(135deg, var(--soft-surface), rgba(47, 159, 216, 0.20));
}

.game-card__media img {
    width: 100%;
    height: 100%;
    min-height: 260px;
    object-fit: cover;
    display: block;
}

.game-card__body {
    min-height: 260px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.game-card__title {
    color: var(--steam-ink) !important;
    margin: 0 0 0.35rem;
    font-size: 1.45rem;
    line-height: 1.15;
    font-weight: 850;
}

.game-card__meta,
.game-card__caption {
    color: var(--steam-muted) !important;
    margin: 0.2rem 0;
    font-size: 0.92rem;
}

.game-card__description {
    color: var(--steam-ink) !important;
    margin: 0.65rem 0 0;
}

.game-card__progress {
    height: 0.7rem;
    border-radius: 999px;
    background: var(--soft-surface);
    overflow: hidden;
    margin-top: 0.8rem;
}

.game-card__progress span {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, var(--steam-green), var(--steam-cyan));
}

.game-card__button {
    display: inline-block;
    align-self: flex-start;
    margin-top: 0.9rem;
    padding: 0.55rem 0.85rem;
    background: linear-gradient(135deg, var(--steam-green), #146849);
    color: #ffffff !important;
    border-radius: 8px;
    text-decoration: none !important;
    font-weight: 800;
}

@media (max-width: 760px) {
    .game-card {
        grid-template-columns: 1fr;
    }

    .game-card__media,
    .game-card__media img,
    .game-card__body {
        min-height: 210px;
    }

    .home-banner__content {
        padding: 1.4rem;
    }
}
</style>
"""
    return (
        css.replace("__INK__", theme["ink"])
        .replace("__MUTED__", theme["muted"])
        .replace("__GREEN__", theme["green"])
        .replace("__CYAN__", theme["cyan"])
        .replace("__AMBER__", theme["amber"])
        .replace("__SURFACE__", theme["surface"])
        .replace("__SOFT_SURFACE__", theme["soft_surface"])
        .replace("__BACKGROUND__", theme["background"])
        .replace("__BORDER__", theme["border"])
        .replace("__SHADOW__", theme["shadow"])
        .replace("__HERO_A__", theme["hero_a"])
        .replace("__HERO_B__", theme["hero_b"])
    )


@st.cache_data(show_spinner=False)
def image_to_data_url(path: str) -> str:
    image_path = Path(path)
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def home_banner(image_path: str):
    image_url = image_to_data_url(image_path)
    st.markdown(
        f"""
        <section class="home-banner has-image" style="background:
            linear-gradient(90deg, rgba(7, 15, 13, 0.90), rgba(7, 15, 13, 0.58), rgba(7, 15, 13, 0.16)),
            url('{image_url}');">
        </section>
        """,
        unsafe_allow_html=True,
    )


def page_setup(title: str):
    try:
        st.set_page_config(
            page_title=f"{title} | Steam Game Recommender",
            page_icon="🎮",
            layout="wide",
        )
    except st.errors.StreamlitAPIException:
        pass
    st.markdown(app_css(), unsafe_allow_html=True)


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
    default_all_columns: bool = False,
):
    row_key = f"{key}_rows" if key else None
    column_key = f"{key}_columns" if key else None
    selected = st.selectbox(label, PREVIEW_OPTIONS, index=1, key=row_key)

    preview_columns = [column for column in (columns or df.columns.tolist()) if column in df.columns]
    if allow_all_columns and columns:
        column_options = (
            ["All columns", "Selected preview columns"]
            if default_all_columns
            else ["Selected preview columns", "All columns"]
        )
        column_mode = st.radio(
            "Columns to show",
            column_options,
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

    st.dataframe(preview_df, width="stretch")
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
    price = "Free" if bool(row.get("is_free", False)) else f"${row.get('price', 0):.2f}"
    name = str(row.get("name", "Unknown game"))
    release_year = int(row.get("release_year", 0) or 0)
    rating = float(row.get("rating_percent", 0) or 0)
    reviews = int(row.get("total_reviews", 0) or 0)
    description = str(row.get("short_description", "")).strip()
    if len(description) > 280:
        description = f"{description[:280]}..."
    image = row.get("header_image", "")
    appid = row.get("appid")

    with st.container(border=True):
        media_col, body_col = st.columns([1, 1.8], gap="large")

        with media_col:
            if isinstance(image, str) and image.startswith("http"):
                st.image(image, width="stretch")
            else:
                st.markdown(
                    '<div class="game-card__media" aria-hidden="true"></div>',
                    unsafe_allow_html=True,
                )

        with body_col:
            st.subheader(name)
            st.caption(
                f"{price} | {release_year} | {rating:.0f}% positive | {reviews:,} reviews"
            )

            if "match_score" in row and pd.notna(row.get("match_score")):
                match_score = float(row["match_score"])
                st.progress(
                    max(0.0, min(match_score / 100, 1.0)),
                    text=f"Match Score: {match_score:.0f}%",
                )

            shared_reasons = row.get("shared_reasons")
            if isinstance(shared_reasons, str) and shared_reasons.strip():
                st.write(f"**Recommended because:** {shared_reasons}")

            score_breakdown = row.get("score_breakdown")
            if isinstance(score_breakdown, str) and score_breakdown.strip():
                st.caption(f"Score mix: {score_breakdown}")

            if description:
                st.write(description)

            if pd.notna(appid):
                st.link_button(
                    "Open on Steam",
                    f"https://store.steampowered.com/app/{int(appid)}",
                )
