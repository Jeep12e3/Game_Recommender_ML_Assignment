from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import build_feature_text


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0) for value in weights.values())
    if total == 0:
        return {"content": 1.0, "rating": 0.0, "popularity": 0.0, "recency": 0.0}
    return {key: max(value, 0) / total for key, value in weights.items()}


@st.cache_resource(show_spinner="Building TF-IDF recommender...")
def build_vector_model(_df: pd.DataFrame, selected_features_key: tuple[tuple[str, bool], ...]):
    selected_features = dict(selected_features_key)
    feature_text = _df.apply(lambda row: build_feature_text(row, selected_features), axis=1)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=35_000,
        min_df=2,
        ngram_range=(1, 2),
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

    weights = normalize_weights(weights)
    popularity_score = (
        0.50 * candidate_scores["total_reviews_scaled"]
        + 0.35 * candidate_scores["owner_midpoint_scaled"]
        + 0.15 * candidate_scores["peak_ccu_scaled"]
    )

    final_score = (
        weights["content"] * content_scores
        + weights["rating"] * candidate_scores["rating_percent_scaled"].to_numpy()
        + weights["popularity"] * popularity_score.to_numpy()
        + weights["recency"] * candidate_scores["release_year_scaled"].to_numpy()
    )

    candidate_scores["content_similarity"] = content_scores
    candidate_scores["match_score"] = np.clip(final_score * 100, 0, 100)
    candidate_scores["shared_reasons"] = candidate_scores.apply(
        lambda row: shared_reasons(df.loc[selected_idx], row),
        axis=1,
    )

    return candidate_scores.sort_values("match_score", ascending=False).head(top_n)


def shared_reasons(source: pd.Series, target: pd.Series, limit: int = 5) -> str:
    source_tokens = set(source.get("genres_list", []) + source.get("tags_list", []))
    target_tokens = set(target.get("genres_list", []) + target.get("tags_list", []))
    shared = list(source_tokens.intersection(target_tokens))
    return ", ".join(shared[:limit]) if shared else "Similar description and metadata"
