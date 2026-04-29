import streamlit as st

from src.config import DEFAULT_FEATURES, DEFAULT_WEIGHTS
from src.data_loader import active_preprocessing_key, load_prepared_games
from src.recommender import build_vector_model, normalize_weights
from src.ui import page_setup, require_data


page_setup("Build Recommender")
st.title("Build Recommender")

df = load_prepared_games()
if not require_data(df):
    st.stop()
data_key = active_preprocessing_key()

st.write(
    "Choose which game information should influence similarity, then set how much the final "
    "match score should care about content, rating, popularity, and recency."
)

st.subheader("Content Features")
selected_features = {}
cols = st.columns(3)
for index, (feature, default) in enumerate(DEFAULT_FEATURES.items()):
    selected_features[feature] = cols[index % 3].checkbox(
        feature.replace("_", " ").title(),
        value=st.session_state.get("selected_features", DEFAULT_FEATURES).get(feature, default),
    )

if not any(selected_features.values()):
    st.warning("Select at least one feature to build the recommender.")
    st.stop()

st.subheader("Score Weights")
left, right = st.columns(2)
current_weights = st.session_state.get("score_weights", DEFAULT_WEIGHTS)
weights = {
    "content": left.slider("Content Similarity", 0.0, 1.0, float(current_weights["content"]), 0.05),
    "rating": left.slider("Rating Score", 0.0, 1.0, float(current_weights["rating"]), 0.05),
    "popularity": right.slider("Popularity Score", 0.0, 1.0, float(current_weights["popularity"]), 0.05),
    "recency": right.slider("Recency Score", 0.0, 1.0, float(current_weights["recency"]), 0.05),
}
normalized = normalize_weights(weights)

st.caption(
    f"Normalized weights: content {normalized['content']:.0%}, rating {normalized['rating']:.0%}, "
    f"popularity {normalized['popularity']:.0%}, recency {normalized['recency']:.0%}."
)

if st.button("Build / Update Recommender", type="primary"):
    st.session_state["selected_features"] = selected_features
    st.session_state["score_weights"] = normalized
    feature_key = tuple(sorted(selected_features.items()))
    build_vector_model(df, feature_key, data_key)
    st.success("Recommender is ready. Go to the Game Recommender page to test it.")

st.subheader("How The Match Score Works")
st.code(
    """match_score =
    content_similarity * content_weight
  + rating_score       * rating_weight
  + popularity_score   * popularity_weight
  + recency_score      * recency_weight""",
    language="text",
)

st.info(
    "The app does not build a huge 90k x 90k similarity matrix. It vectorizes the games and calculates "
    "similarity only when recommendations are requested, which is lighter for Streamlit."
)
