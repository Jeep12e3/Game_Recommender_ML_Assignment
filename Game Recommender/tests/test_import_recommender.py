def test_import_recommender():
    from src.recommender import normalize_weights

    assert normalize_weights({"content": 1}) == {"content": 1.0}
