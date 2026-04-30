from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    BASE_CLEAN_PARQUET,
    BASE_CLEAN_META,
    CSV_INSIDE_ZIP,
    ENV_DATA_PATH,
    LOCAL_DATA_PATH_FILE,
    PRIMARY_CSV,
    PRIMARY_ZIP,
    PROCESSED_PARQUET,
    PROCESSED_META,
)
from src.preprocessing import (
    apply_preprocessing_options,
    base_clean_games,
    normalize_preprocessing_options,
    preprocessing_options_key,
)


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


@st.cache_data(show_spinner="Base-cleaning Steam dataset...")
def _base_clean_games_cached(_raw_df: pd.DataFrame, data_signature: tuple) -> pd.DataFrame:
    return base_clean_games(_raw_df)


def _data_signature(df: pd.DataFrame) -> tuple:
    if df.empty:
        return (0, 0, None)
    appid_max = df["appid"].max() if "appid" in df else None
    return (len(df), len(df.columns), appid_max)


def _read_cache_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_cache_meta(path: Path, meta: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(meta), indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _cache_matches(path: Path, expected: dict) -> bool:
    meta = _read_cache_meta(path)
    expected = _jsonable(expected)
    return all(meta.get(key) == value for key, value in expected.items())


def load_base_clean_games() -> pd.DataFrame:
    raw_df = load_games()
    if raw_df.empty:
        return raw_df

    raw_signature = _data_signature(raw_df)
    base_meta = {"raw_signature": raw_signature}
    if BASE_CLEAN_PARQUET.exists() and _cache_matches(BASE_CLEAN_META, base_meta):
        return pd.read_parquet(BASE_CLEAN_PARQUET)

    base_df = _base_clean_games_cached(raw_df, raw_signature)
    try:
        BASE_CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        base_df.to_parquet(BASE_CLEAN_PARQUET, index=False)
        _write_cache_meta(BASE_CLEAN_META, base_meta)
    except (ImportError, OSError, ValueError):
        pass
    return base_df


@st.cache_data(show_spinner="Applying preprocessing options...")
def _apply_preprocessing_cached(_base_df: pd.DataFrame, options_key: tuple, data_signature: tuple) -> pd.DataFrame:
    options = dict(options_key)
    return apply_preprocessing_options(_base_df, **options)


def load_prepared_games(options: dict | None = None) -> pd.DataFrame:
    raw_df = load_games()
    base_df = load_base_clean_games()
    if base_df.empty:
        return base_df

    active_options = normalize_preprocessing_options(
        options or st.session_state.get("preprocessing_options"),
        raw_df,
    )
    options_key = preprocessing_options_key(active_options, raw_df)
    default_key = preprocessing_options_key(None, raw_df)
    processed_meta = {
        "base_signature": _data_signature(base_df),
        "options_key": options_key,
    }

    if (
        options_key == default_key
        and PROCESSED_PARQUET.exists()
        and _cache_matches(PROCESSED_META, processed_meta)
    ):
        return pd.read_parquet(PROCESSED_PARQUET)

    prepared = _apply_preprocessing_cached(base_df, options_key, _data_signature(base_df))
    if options_key == default_key:
        try:
            PROCESSED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
            prepared.to_parquet(PROCESSED_PARQUET, index=False)
            _write_cache_meta(PROCESSED_META, processed_meta)
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
    _base_clean_games_cached.clear()
    _apply_preprocessing_cached.clear()
    if BASE_CLEAN_PARQUET.exists():
        BASE_CLEAN_PARQUET.unlink()
    if BASE_CLEAN_META.exists():
        BASE_CLEAN_META.unlink()
    if PROCESSED_PARQUET.exists():
        PROCESSED_PARQUET.unlink()
    if PROCESSED_META.exists():
        PROCESSED_META.unlink()


def data_help_message() -> str:
    return (
        "Place `games_march2025_cleaned.csv` inside the `data/` folder. "
        "Alternatively, place the Kaggle zip as `data/archive (2).zip`; the app will read "
        "`games_march2025_cleaned.csv` from inside it."
    )
