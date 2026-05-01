from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

from src.config import (
    BASE_CLEAN_SCHEMA_VERSION,
    BASE_CLEAN_PARQUET,
    BASE_CLEAN_META,
    CSV_INSIDE_ZIP,
    ENV_DATA_PATH,
    LOCAL_DATA_PATH_FILE,
    PRIMARY_CSV,
    PRIMARY_ZIP,
    PROCESSED_PARQUET,
    PROCESSED_META,
    PROCESSED_SCHEMA_VERSION,
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


BASE_REQUIRED_COLUMNS = {
    "release_year",
    "is_free",
    "total_reviews",
    "rating_percent",
}

PROCESSED_REQUIRED_COLUMNS = {
    "rating_percent_scaled",
    "total_reviews_scaled",
    "owner_midpoint_scaled",
    "peak_ccu_scaled",
    "release_year_scaled",
}


def _normalize_platform_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int).astype(bool)

    lowered = series.fillna("").astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y"}
    return lowered.isin(truthy)


def _normalize_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in ("windows", "mac", "linux"):
        if column in normalized.columns:
            normalized[column] = _normalize_platform_column(normalized[column])
    return normalized


@st.cache_data(show_spinner="Loading Steam dataset...")
def load_games() -> pd.DataFrame:
    source_type, source_path = available_data_source()

    if source_path is None:
        return pd.DataFrame()

    if source_type == "csv":
        return _normalize_raw_df(pd.read_csv(source_path, usecols=lambda col: col in IMPORTANT_COLUMNS))

    with zipfile.ZipFile(source_path) as archive:
        if CSV_INSIDE_ZIP not in archive.namelist():
            raise FileNotFoundError(f"{CSV_INSIDE_ZIP} was not found inside {source_path.name}")
        with archive.open(CSV_INSIDE_ZIP) as file:
            return _normalize_raw_df(pd.read_csv(file, usecols=lambda col: col in IMPORTANT_COLUMNS))


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _cache_matches(path: Path, expected: dict) -> bool:
    meta = _read_cache_meta(path)
    expected = _jsonable(expected)
    return all(meta.get(key) == value for key, value in expected.items())


def _cache_has_columns(path: Path, required_columns: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        cached_df = pd.read_parquet(path, columns=list(required_columns))
    except (OSError, ValueError, KeyError, FileNotFoundError):
        return False
    return required_columns.issubset(cached_df.columns)


def load_base_clean_games() -> pd.DataFrame:
    raw_df = load_games()
    if raw_df.empty:
        return raw_df

    raw_signature = _data_signature(raw_df)
    base_meta = {"raw_signature": raw_signature, "schema_version": BASE_CLEAN_SCHEMA_VERSION}
    if (
        BASE_CLEAN_PARQUET.exists()
        and _cache_matches(BASE_CLEAN_META, base_meta)
        and _cache_has_columns(BASE_CLEAN_PARQUET, BASE_REQUIRED_COLUMNS)
    ):
        return pd.read_parquet(BASE_CLEAN_PARQUET)

    base_df = _base_clean_games_cached(raw_df, raw_signature)
    try:
        BASE_CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        base_df.to_parquet(BASE_CLEAN_PARQUET, index=False)
        _write_cache_meta(BASE_CLEAN_META, base_meta)
    except (ImportError, OSError, ValueError):
        pass
    return base_df


def _apply_preprocessing_cached(_base_df: pd.DataFrame, options_key: tuple, data_signature: tuple) -> pd.DataFrame:
    options = normalize_preprocessing_options(dict(options_key))
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
        "schema_version": PROCESSED_SCHEMA_VERSION,
    }

    if (
        options_key == default_key
        and PROCESSED_PARQUET.exists()
        and _cache_matches(PROCESSED_META, processed_meta)
        and _cache_has_columns(PROCESSED_PARQUET, PROCESSED_REQUIRED_COLUMNS)
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
