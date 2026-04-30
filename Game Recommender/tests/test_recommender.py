import pandas as pd

from src.config import DEFAULT_FEATURES, DEFAULT_WEIGHTS
from src.recommender import build_vector_model, normalize_weights, recommend_games, shared_reasons


def _row(
    name,
    genres,
    tags,
    description,
    rating_scaled,
    reviews_scaled,
    owners_scaled,
    ccu_scaled,
    year_scaled,
):
    return {
        "appid": hash(name) % 100000,
        "name": name,
        "genres_list": genres,
        "tags_list": tags,
        "categories_list": ["Single-player"],
        "genres_text": " ".join(genres).lower(),
        "tags_text": " ".join(tags).lower(),
        "categories_text": "single_player",
        "developers_text": "",
        "publishers_text": "",
        "short_description": description,
        "rating_percent_scaled": rating_scaled,
        "total_reviews_scaled": reviews_scaled,
        "owner_midpoint_scaled": owners_scaled,
        "peak_ccu_scaled": ccu_scaled,
        "release_year_scaled": year_scaled,
        "windows": True,
        "mac": False,
        "linux": False,
        "is_free": False,
        "price": 9.99,
        "release_year": 2024,
        "rating_percent": 90,
        "total_reviews": int(reviews_scaled * 100000),
    }


def test_normalize_weights_falls_back_to_content_only():
    assert normalize_weights({"content": 0, "rating": 0, "popularity": 0, "recency": 0}) == {
        "content": 1.0,
        "rating": 0.0,
        "popularity": 0.0,
        "recency": 0.0,
    }


def test_shared_reasons_canonicalizes_duplicate_tag_casing():
    source = pd.Series({"genres_list": ["Action"], "tags_list": ["Free to Play"], "categories_list": []})
    target = pd.Series({"genres_list": ["Action"], "tags_list": ["Free To Play"], "categories_list": []})

    reasons = shared_reasons(source, target)

    assert "Action" in reasons
    assert reasons.count("Free To Play") == 1


def test_recommendation_prefers_similar_game_over_unrelated_popular_game():
    df = pd.DataFrame(
        [
            _row(
                "Space Quest",
                ["Adventure", "RPG"],
                ["Story Rich", "Sci-fi", "Exploration"],
                "Narrative space adventure with exploration and role playing choices.",
                0.90,
                0.20,
                0.20,
                0.10,
                0.95,
            ),
            _row(
                "Space Quest Echoes",
                ["Adventure", "RPG"],
                ["Story Rich", "Sci-fi", "Exploration"],
                "Narrative space adventure with exploration and role playing choices.",
                0.82,
                0.12,
                0.12,
                0.05,
                0.90,
            ),
            _row(
                "Stadium Champion",
                ["Sports"],
                ["Football", "Competitive", "Multiplayer"],
                "Fast competitive football action in packed stadiums.",
                0.98,
                1.00,
                1.00,
                1.00,
                0.99,
            ),
        ]
    )
    feature_key = tuple(sorted(DEFAULT_FEATURES.items()))
    _, matrix = build_vector_model(df, feature_key, ("test",))

    recommendations = recommend_games(df, df, "Space Quest", matrix, DEFAULT_WEIGHTS, top_n=2)

    assert recommendations.iloc[0]["name"] == "Space Quest Echoes"
    assert recommendations.iloc[0]["match_score"] > recommendations.iloc[1]["match_score"]
