from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import CSV_INSIDE_ZIP, ENV_DATA_PATH, LOCAL_DATA_PATH_FILE, PRIMARY_CSV, PRIMARY_ZIP


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


def data_help_message() -> str:
    return (
        "Place `games_march2025_cleaned.csv` inside the `data/` folder. "
        "Alternatively, place the Kaggle zip as `data/archive (2).zip`; the app will read "
        "`games_march2025_cleaned.csv` from inside it."
    )
