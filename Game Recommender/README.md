# Steam Game Recommender

A Streamlit game recommender system using a recent Steam games dataset.

The app recommends games with a content-based approach using genres, Steam tags, categories, descriptions, and optional metadata such as developer and publisher. It also adds a match score using rating, popularity, and release recency signals.

## Pages

1. **Home**  
   Project overview, dataset summary, and workflow.

2. **Dataset & Preprocessing**  
   Shows raw/prepared data, cleaning steps, parsed fields, review features, and MinMaxScaler outputs.

3. **EDA / Insights**  
   Visualizes release trends, top genres/tags, review distribution, popular games, platforms, and free vs paid games.

4. **Build Recommender**  
   Lets users choose the features and weights used by the recommender.

5. **Game Recommender**  
   Lets users select a game, apply filters, and receive recommendations with match scores and explanations.

## Dataset

Use the Kaggle dataset:

[Steam Games Dataset 2025 by Artemiy Ermilov](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)

Recommended file:

```text
games_march2025_cleaned.csv
```

Place it here:

```text
data/games_march2025_cleaned.csv
```

Alternatively, place the downloaded zip here:

```text
data/archive (2).zip
```

You can also point the app to a local CSV/ZIP without copying it:

```bash
set STEAM_DATA_PATH=C:\path\to\archive (2).zip
streamlit run app.py
```

Another local-only option is to create:

```text
data/local_data_path.txt
```

and put the full path to your CSV/ZIP inside it. This file is ignored by Git.

The large dataset files are ignored by GitHub through `.gitignore`.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## How Recommendations Work

The app:

1. Cleans and prepares the dataset.
2. Combines selected metadata fields into a text feature.
3. Converts game text into TF-IDF vectors.
4. Computes cosine similarity for the selected game.
5. Adds genre/tag/category overlap so recommendations stay closer to the selected game's actual taste profile.
6. Blends content similarity with normalized numeric signals:
   - rating score
   - popularity score
   - recency score

The final score is shown as a match score:

```text
match_score =
    content_similarity * content_weight
  + rating_score       * rating_weight
  + popularity_score   * popularity_weight
  + recency_score      * recency_weight
```

Popularity is dampened so very famous but unrelated games do not overpower closer niche matches. Recommendation cards also show shared reasons and a compact score breakdown.

Processed parquet caches are paired with metadata files so stale caches are rebuilt when the source data or preprocessing settings change.

## Notes For Collaborators

- Do not commit the dataset CSV/ZIP files.
- Keep new shared logic inside `src/`.
- Keep Streamlit page-specific UI inside `app.py` or `pages/`.
- If recommendation performance becomes slow, reduce `max_features` in `src/recommender.py`.
- Run checks with `pytest -q`.
