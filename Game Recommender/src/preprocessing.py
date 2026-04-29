from __future__ import annotations

import ast
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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


def prepare_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    prepared = df.copy()
    prepared = prepared.drop_duplicates(subset=["appid"]).drop_duplicates(subset=["name"])

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
    prepared["tags_list"] = prepared.get("tags", "").apply(parse_tags)
    prepared["developers_list"] = prepared.get("developers", "").apply(parse_list_text)
    prepared["publishers_list"] = prepared.get("publishers", "").apply(parse_list_text)

    prepared["genres_text"] = prepared["genres_list"].apply(_join)
    prepared["categories_text"] = prepared["categories_list"].apply(_join)
    prepared["tags_text"] = prepared["tags_list"].apply(_join)
    prepared["developers_text"] = prepared["developers_list"].apply(_join)
    prepared["publishers_text"] = prepared["publishers_list"].apply(_join)
    prepared["short_description"] = prepared.get("short_description", "").fillna("")

    scale_columns = ["rating_percent", "total_reviews", "owner_midpoint", "peak_ccu", "release_year"]
    scaler = MinMaxScaler()
    prepared[[f"{col}_scaled" for col in scale_columns]] = scaler.fit_transform(prepared[scale_columns])

    return prepared.reset_index(drop=True)


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
