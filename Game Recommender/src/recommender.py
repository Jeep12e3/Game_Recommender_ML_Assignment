from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import build_feature_text


CANONICAL_TAGS = {
    "free to play": "Free To Play",
    "freetoplay": "Free To Play",
    "rpg": "RPG",
    "mmorpg": "MMORPG",
    "moba": "MOBA",
    "vr": "VR",
}


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0) for value in weights.values())
    if total == 0:
        return {"content": 1.0, "rating": 0.0, "popularity": 0.0, "recency": 0.0}
    return {key: max(value, 0) / total for key, value in weights.items()}


@st.cache_resource(show_spinner="Building TF-IDF recommender...")
def build_vector_model(_df: pd.DataFrame, selected_features_key: tuple[tuple[str, bool], ...], data_key: tuple = ()):
    selected_features = dict(selected_features_key)
    feature_text = _df.apply(lambda row: build_feature_text(row, selected_features), axis=1)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=35_000,
        min_df=1 if len(_df) < 50 else 2,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(feature_text)
    return vectorizer, matrix


def recommend_games(
    df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    selected_game: str,
    matrix,
    weights: dict[str, float],
    top_n: int = 10,
) -> pd.DataFrame:
    if df.empty or candidate_df.empty or selected_game not in set(df["name"]):
        return pd.DataFrame()

    selected_idx = int(df.index[df["name"].eq(selected_game)][0])
    candidate_indices = candidate_df.index[candidate_df["name"].ne(selected_game)].to_numpy()
    if len(candidate_indices) == 0:
        return pd.DataFrame()

    content_scores = cosine_similarity(matrix[selected_idx], matrix[candidate_indices]).ravel()
    candidate_scores = df.loc[candidate_indices].copy()
    source = df.loc[selected_idx]

    weights = normalize_weights(weights)
    popularity_score = (
        0.45 * dampen_popularity(candidate_scores["total_reviews_scaled"])
        + 0.35 * dampen_popularity(candidate_scores["owner_midpoint_scaled"])
        + 0.20 * dampen_popularity(candidate_scores["peak_ccu_scaled"])
    )
    overlap_scores = candidate_scores.apply(lambda row: overlap_score(source, row), axis=1).to_numpy()
    platform_scores = candidate_scores.apply(lambda row: platform_score(source, row), axis=1).to_numpy()
    quality_scores = quality_score(candidate_scores).to_numpy()

    content_signal = (0.80 * content_scores) + (0.20 * overlap_scores)

    final_score = (
        weights["content"] * content_signal
        + weights["rating"] * candidate_scores["rating_percent_scaled"].to_numpy()
        + weights["popularity"] * popularity_score.to_numpy()
        + weights["recency"] * candidate_scores["release_year_scaled"].to_numpy()
    )
    final_score = (0.90 * final_score) + (0.06 * quality_scores) + (0.04 * platform_scores)

    candidate_scores["content_similarity"] = content_scores
    candidate_scores["overlap_score"] = overlap_scores
    candidate_scores["quality_score"] = quality_scores
    candidate_scores["popularity_score"] = popularity_score
    candidate_scores["match_score"] = np.clip(final_score * 100, 0, 100)
    candidate_scores["shared_reasons"] = candidate_scores.apply(
        lambda row: shared_reasons(source, row),
        axis=1,
    )
    candidate_scores["score_breakdown"] = candidate_scores.apply(
        lambda row: score_breakdown(row, weights),
        axis=1,
    )

    return candidate_scores.sort_values("match_score", ascending=False).head(top_n)


def _as_token_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist() if str(item).strip()]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    return [str(value)] if str(value).strip() else []


def _canonical_token(value: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value).replace("_", " ").strip())
    lookup = re.sub(r"[^a-z0-9]+", "", normalized.lower())
    if lookup in CANONICAL_TAGS:
        return CANONICAL_TAGS[lookup]
    return normalized.title() if normalized.islower() else normalized


def token_set(row: pd.Series, columns: tuple[str, ...]) -> set[str]:
    tokens: set[str] = set()
    for column in columns:
        tokens.update(_canonical_token(item) for item in _as_token_list(row.get(column)))
    return {token for token in tokens if token}


def overlap_score(source: pd.Series, target: pd.Series) -> float:
    source_genres = token_set(source, ("genres_list",))
    target_genres = token_set(target, ("genres_list",))
    source_tags = token_set(source, ("tags_list", "categories_list"))
    target_tags = token_set(target, ("tags_list", "categories_list"))

    genre_overlap = len(source_genres & target_genres) / max(len(source_genres), 1)
    tag_overlap = len(source_tags & target_tags) / max(len(source_tags), 1)
    return float(np.clip((0.60 * genre_overlap) + (0.40 * tag_overlap), 0, 1))


def platform_score(source: pd.Series, target: pd.Series) -> float:
    platforms = ("windows", "mac", "linux")
    source_platforms = {platform for platform in platforms if bool(source.get(platform, False))}
    target_platforms = {platform for platform in platforms if bool(target.get(platform, False))}
    if not source_platforms:
        return 0.0
    return len(source_platforms & target_platforms) / len(source_platforms)


def dampen_popularity(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0, upper=1)
    return np.sqrt(values)


def quality_score(df: pd.DataFrame) -> pd.Series:
    rating = pd.to_numeric(df["rating_percent_scaled"], errors="coerce").fillna(0)
    reviews = dampen_popularity(df["total_reviews_scaled"])
    return (0.70 * rating) + (0.30 * reviews)


def shared_reasons(source: pd.Series, target: pd.Series, limit: int = 5) -> str:
    source_tokens = token_set(source, ("genres_list", "tags_list", "categories_list"))
    target_tokens = token_set(target, ("genres_list", "tags_list", "categories_list"))
    shared = sorted(source_tokens.intersection(target_tokens))
    return ", ".join(shared[:limit]) if shared else "Similar description and metadata"


def score_breakdown(row: pd.Series, weights: dict[str, float]) -> str:
    content = row.get("content_similarity", 0) * weights["content"] * 100
    rating = row.get("rating_percent_scaled", 0) * weights["rating"] * 100
    popularity = row.get("popularity_score", 0) * weights["popularity"] * 100
    recency = row.get("release_year_scaled", 0) * weights["recency"] * 100
    return (
        f"Content {content:.0f}, rating {rating:.0f}, "
        f"popularity {popularity:.0f}, recency {recency:.0f}"
    )
