from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import CSV_INSIDE_ZIP, ENV_DATA_PATH, LOCAL_DATA_PATH_FILE, PRIMARY_CSV, PRIMARY_ZIP, PROCESSED_PARQUET
from src.preprocessing import normalize_preprocessing_options, prepare_games, preprocessing_options_key


IMPORTANT_COLUMNS = [
    "appid",
    "name",
    "release_date",
    "required_age",
    "price",
    "dlc_count",
    "short_description",
    "header_image",
    "windows",
    "mac",
    "linux",
    "metacritic_score",
    "achievements",
    "recommendations",
    "developers",
    "publishers",
    "categories",
    "genres",
    "positive",
    "negative",
    "estimated_owners",
    "average_playtime_forever",
    "median_playtime_forever",
    "discount",
    "peak_ccu",
    "tags",
    "pct_pos_total",
    "num_reviews_total",
    "pct_pos_recent",
    "num_reviews_recent",
]


def available_data_source() -> tuple[str, Path | None]:
    if ENV_DATA_PATH:
        env_path = Path(ENV_DATA_PATH)
        if env_path.exists() and env_path.suffix.lower() == ".csv":
            return "csv", env_path
        if env_path.exists() and env_path.suffix.lower() == ".zip":
            return "zip", env_path
    if LOCAL_DATA_PATH_FILE.exists():
        local_path = Path(LOCAL_DATA_PATH_FILE.read_text(encoding="utf-8").strip())
        if local_path.exists() and local_path.suffix.lower() == ".csv":
            return "csv", local_path
        if local_path.exists() and local_path.suffix.lower() == ".zip":
            return "zip", local_path
    if PRIMARY_CSV.exists():
        return "csv", PRIMARY_CSV
    if PRIMARY_ZIP.exists():
        return "zip", PRIMARY_ZIP
    return "missing", None


@st.cache_data(show_spinner="Loading Steam dataset...")
def load_games() -> pd.DataFrame:
    source_type, source_path = available_data_source()

    if source_path is None:
        return pd.DataFrame()

    if source_type == "csv":
        return pd.read_csv(source_path, usecols=lambda col: col in IMPORTANT_COLUMNS)

    with zipfile.ZipFile(source_path) as archive:
        if CSV_INSIDE_ZIP not in archive.namelist():
            raise FileNotFoundError(f"{CSV_INSIDE_ZIP} was not found inside {source_path.name}")
        with archive.open(CSV_INSIDE_ZIP) as file:
            return pd.read_csv(file, usecols=lambda col: col in IMPORTANT_COLUMNS)


@st.cache_data(show_spinner="Preparing Steam dataset...")
def _prepare_games_cached(_raw_df: pd.DataFrame, options_key: tuple) -> pd.DataFrame:
    options = dict(options_key)
    return prepare_games(_raw_df, **options)


def load_prepared_games(options: dict | None = None) -> pd.DataFrame:
    raw_df = load_games()
    if raw_df.empty:
        return raw_df

    active_options = normalize_preprocessing_options(
        options or st.session_state.get("preprocessing_options"),
        raw_df,
    )
    options_key = preprocessing_options_key(active_options, raw_df)
    default_key = preprocessing_options_key(None, raw_df)

    if options_key == default_key and PROCESSED_PARQUET.exists():
        return pd.read_parquet(PROCESSED_PARQUET)

    prepared = _prepare_games_cached(raw_df, options_key)
    if options_key == default_key:
        try:
            PROCESSED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
            prepared.to_parquet(PROCESSED_PARQUET, index=False)
        except (ImportError, OSError, ValueError):
            pass
    return prepared


def save_active_preprocessing_options(options: dict) -> None:
    raw_df = load_games()
    st.session_state["preprocessing_options"] = normalize_preprocessing_options(options, raw_df)


def active_preprocessing_key() -> tuple:
    raw_df = load_games()
    return preprocessing_options_key(st.session_state.get("preprocessing_options"), raw_df)


def save_processed_download_cache(prepared: pd.DataFrame) -> None:
    try:
        PROCESSED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_parquet(PROCESSED_PARQUET, index=False)
    except (ImportError, OSError, ValueError):
        pass


def clear_prepared_cache() -> None:
    load_games.clear()
    _prepare_games_cached.clear()
    if PROCESSED_PARQUET.exists():
        PROCESSED_PARQUET.unlink()


def data_help_message() -> str:
    return (
        "Place `games_march2025_cleaned.csv` inside the `data/` folder. "
        "Alternatively, place the Kaggle zip as `data/archive (2).zip`; the app will read "
        "`games_march2025_cleaned.csv` from inside it."
    )
