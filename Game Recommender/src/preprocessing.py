from __future__ import annotations

import ast
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


MATURE_CONTENT_TERMS = {
    "adult content",
    "sexual content",
    "nudity",
    "hentai",
    "nsfw",
    "porn",
    "pornographic",
    "explicit sexual content",
    "adult only",
    "adults only",
}


def _safe_literal(value):
    if pd.isna(value):
        return []
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def parse_list_text(value) -> list[str]:
    parsed = _safe_literal(value)
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str):
        return [part.strip() for part in re.split(r"[;,]", parsed) if part.strip()]
    return []


def parse_tags(value, limit: int = 20) -> list[str]:
    parsed = _safe_literal(value)
    if isinstance(parsed, dict):
        sorted_tags = sorted(parsed.items(), key=lambda item: item[1], reverse=True)
        return [str(tag).strip() for tag, _ in sorted_tags[:limit] if str(tag).strip()]
    if isinstance(parsed, list):
        return [str(tag).strip() for tag in parsed[:limit] if str(tag).strip()]
    if isinstance(parsed, str):
        return [part.strip() for part in re.split(r"[;,]", parsed)[:limit] if part.strip()]
    return []


def _join(values: Iterable[str]) -> str:
    return " ".join(str(value).replace(" ", "_").lower() for value in values if str(value).strip())


def parse_owner_midpoint(value) -> float:
    if pd.isna(value):
        return 0.0
    numbers = [int(num) for num in re.findall(r"\d+", str(value).replace(",", ""))]
    if len(numbers) >= 2:
        return float(sum(numbers[:2]) / 2)
    if len(numbers) == 1:
        return float(numbers[0])
    return 0.0


def is_mature_content(row: pd.Series) -> bool:
    text_parts = [
        row.get("name", ""),
        row.get("short_description", ""),
        row.get("genres", ""),
        row.get("categories", ""),
        row.get("tags", ""),
    ]
    parsed_parts = []
    parsed_parts.extend(parse_list_text(row.get("genres", "")))
    parsed_parts.extend(parse_list_text(row.get("categories", "")))
    parsed_parts.extend(parse_tags(row.get("tags", ""), limit=50))
    text_parts.extend(parsed_parts)
    combined = " ".join(str(part).lower() for part in text_parts if str(part).strip())
    return any(term in combined for term in MATURE_CONTENT_TERMS)


def count_mature_content_games(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    deduped = df.drop_duplicates(subset=["appid"]).drop_duplicates(subset=["name"])
    return int(deduped.apply(is_mature_content, axis=1).sum())


def default_preprocessing_options(df: pd.DataFrame | None = None) -> dict:
    year_range = None
    if df is not None and not df.empty and "release_date" in df:
        years = pd.to_datetime(df["release_date"], errors="coerce").dt.year.dropna()
        if not years.empty:
            year_range = (int(years.min()), int(years.max()))

    return {
        "remove_duplicates": True,
        "missing_descriptions": "Keep as empty",
        "min_reviews": 0,
        "min_rating": 0,
        "year_range": year_range,
        "platforms": ("windows", "mac", "linux"),
        "price_type": "All",
        "tag_limit": 20,
    }


def normalize_preprocessing_options(options: dict | None, df: pd.DataFrame | None = None) -> dict:
    normalized = default_preprocessing_options(df)
    if options:
        normalized.update(options)

    if isinstance(normalized.get("platforms"), list):
        normalized["platforms"] = tuple(normalized["platforms"])
    if normalized.get("year_range") is not None:
        normalized["year_range"] = tuple(normalized["year_range"])
    return normalized


def preprocessing_options_key(options: dict | None, df: pd.DataFrame | None = None) -> tuple:
    normalized = normalize_preprocessing_options(options, df)
    return tuple(sorted(normalized.items()))


def base_clean_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    prepared = df.copy()
    prepared["release_date"] = pd.to_datetime(prepared["release_date"], errors="coerce")
    prepared["release_year"] = prepared["release_date"].dt.year.fillna(0).astype(int)
    prepared["price"] = pd.to_numeric(prepared.get("price", 0), errors="coerce").fillna(0)
    prepared["is_free"] = prepared["price"].eq(0)

    for column in ["positive", "negative", "num_reviews_total", "pct_pos_total", "peak_ccu"]:
        if column in prepared:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0)

    prepared["total_reviews"] = prepared.get("num_reviews_total", 0)
    if "positive" in prepared and "negative" in prepared:
        fallback_total = prepared["positive"] + prepared["negative"]
        prepared["total_reviews"] = prepared["total_reviews"].where(prepared["total_reviews"].gt(0), fallback_total)
        prepared["rating_percent"] = prepared.get("pct_pos_total", 0).where(
            prepared.get("pct_pos_total", 0).gt(0),
            np.where(fallback_total.gt(0), prepared["positive"] / fallback_total * 100, 0),
        )
    else:
        prepared["rating_percent"] = prepared.get("pct_pos_total", 0)

    prepared["owner_midpoint"] = prepared.get("estimated_owners", "").apply(parse_owner_midpoint)

    prepared["genres_list"] = prepared.get("genres", "").apply(parse_list_text)
    prepared["categories_list"] = prepared.get("categories", "").apply(parse_list_text)
    prepared["tags_list"] = prepared.get("tags", "").apply(lambda value: parse_tags(value, limit=50))
    prepared["developers_list"] = prepared.get("developers", "").apply(parse_list_text)
    prepared["publishers_list"] = prepared.get("publishers", "").apply(parse_list_text)

    mature_mask = prepared.apply(is_mature_content, axis=1)
    prepared = prepared.loc[~mature_mask].copy()
    prepared["short_description"] = prepared.get("short_description", "").fillna("")
    prepared["genres_text"] = prepared["genres_list"].apply(_join)
    prepared["categories_text"] = prepared["categories_list"].apply(_join)
    prepared["developers_text"] = prepared["developers_list"].apply(_join)
    prepared["publishers_text"] = prepared["publishers_list"].apply(_join)

    return prepared.reset_index(drop=True)


def apply_preprocessing_options(
    base_df: pd.DataFrame,
    remove_duplicates: bool = True,
    missing_descriptions: str = "Keep as empty",
    min_reviews: int = 0,
    min_rating: int = 0,
    year_range: tuple[int, int] | None = None,
    platforms: tuple[str, ...] = ("windows", "mac", "linux"),
    price_type: str = "All",
    tag_limit: int = 20,
) -> pd.DataFrame:
    if base_df.empty:
        return base_df

    prepared = base_df.copy()

    if remove_duplicates:
        prepared = prepared.drop_duplicates(subset=["appid"]).drop_duplicates(subset=["name"])

    if missing_descriptions == "Remove missing descriptions":
        prepared = prepared[prepared["short_description"].fillna("").str.strip().ne("")]

    if year_range is not None:
        prepared = prepared[prepared["release_year"].between(year_range[0], year_range[1])]

    valid_platforms = [platform for platform in platforms if platform in {"windows", "mac", "linux"}]
    if valid_platforms:
        platform_mask = prepared[valid_platforms].astype(bool).any(axis=1)
        prepared = prepared[platform_mask]

    if price_type == "Free only":
        prepared = prepared[prepared["is_free"]]
    elif price_type == "Paid only":
        prepared = prepared[~prepared["is_free"]]

    prepared = prepared[prepared["total_reviews"].ge(min_reviews)]
    prepared = prepared[prepared["rating_percent"].ge(min_rating)]

    prepared["tags_list"] = prepared["tags_list"].apply(lambda tags: list(tags)[:tag_limit])
    prepared["tags_text"] = prepared["tags_list"].apply(_join)

    scale_columns = ["rating_percent", "total_reviews", "owner_midpoint", "peak_ccu", "release_year"]
    if prepared.empty:
        for col in scale_columns:
            prepared[f"{col}_scaled"] = []
    else:
        scaler = MinMaxScaler()
        prepared[[f"{col}_scaled" for col in scale_columns]] = scaler.fit_transform(prepared[scale_columns])

    return prepared.reset_index(drop=True)


def prepare_games(df: pd.DataFrame, **options) -> pd.DataFrame:
    return apply_preprocessing_options(base_clean_games(df), **options)


def build_feature_text(row: pd.Series, selected_features: dict[str, bool]) -> str:
    parts = []
    if selected_features.get("genres"):
        parts.append(row.get("genres_text", ""))
    if selected_features.get("tags"):
        parts.append(row.get("tags_text", ""))
    if selected_features.get("categories"):
        parts.append(row.get("categories_text", ""))
    if selected_features.get("short_description"):
        parts.append(str(row.get("short_description", "")).lower())
    if selected_features.get("developers"):
        parts.append(row.get("developers_text", ""))
    if selected_features.get("publishers"):
        parts.append(row.get("publishers_text", ""))
    return " ".join(part for part in parts if part)


def apply_filters(
    df: pd.DataFrame,
    platform: str = "Any",
    genre: str = "Any",
    price_type: str = "Any",
    year_range: tuple[int, int] | None = None,
    min_rating: int = 0,
    min_reviews: int = 0,
) -> pd.DataFrame:
    filtered = df.copy()

    if platform in {"windows", "mac", "linux"}:
        filtered = filtered[filtered[platform].astype(bool)]

    if genre != "Any":
        filtered = filtered[filtered["genres_list"].apply(lambda genres: genre in genres)]

    if price_type == "Free":
        filtered = filtered[filtered["is_free"]]
    elif price_type == "Paid":
        filtered = filtered[~filtered["is_free"]]

    if year_range is not None:
        filtered = filtered[filtered["release_year"].between(year_range[0], year_range[1])]

    filtered = filtered[filtered["rating_percent"].ge(min_rating)]
    filtered = filtered[filtered["total_reviews"].ge(min_reviews)]
    return filtered
