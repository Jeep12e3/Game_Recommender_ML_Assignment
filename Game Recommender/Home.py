import streamlit as st

from src.config import ROOT_DIR
from src.ui import home_banner, page_setup


page_setup("Home")

home_banner(str(ROOT_DIR / "src" / "img" / "steam_game.jpg"))

st.title("Steam Game Recommender")

st.markdown(
    """
    Steam has a very large game catalog, which makes discovery difficult for users who
    already know what kind of games they enjoy but do not know what to try next. A player
    may like a specific game because of its genre, tags, category, gameplay theme, or short
    description, but manually comparing thousands of games is not practical.

    This application introduces a content-based recommendation system for Steam games.
    Instead of asking users to rate many games first, the app starts from one selected game
    and searches for other games with similar metadata. The project is organized as a simple
    machine learning workflow: explore the raw dataset, clean and prepare the data, build
    the recommender model, then generate recommendations with readable match explanations.
    """
)

st.divider()

st.subheader("Project Workflow")

st.markdown(
    """
    **1. Exploratory Data Analysis**

    The Raw Data EDA page is used to understand the original Steam dataset before any
    cleaning is applied. It shows dataset size, missing values, data types, release trends,
    genre and tag distributions, review patterns, platform support, price distribution, and
    an explorer for inspecting any raw column. This step helps identify data quality issues
    before preprocessing.
    """
)
st.page_link("pages/1_Raw_Data_EDA.py", label="Open Raw Data EDA")

st.markdown(
    """
    **2. Data Preprocessing**

    The Preprocessing page prepares the raw dataset so it can be used by the recommender.
    It handles duplicate games, missing descriptions, review and rating filters, release
    year filtering, platform filtering, price filtering, mature-content removal, tag limits,
    and numeric feature scaling. The output of this step is the prepared dataset used by
    the recommendation pages.
    """
)
st.page_link("pages/2_Preprocessing.py", label="Open Preprocessing")

st.markdown(
    """
    **3. Build Recommender Model**

    The Build Recommender page lets users choose which game features should influence
    similarity, such as genres, tags, categories, short descriptions, developers, and
    publishers. It also lets users adjust the score weights for content similarity, rating,
    popularity, and recency before building the vector model.
    """
)
st.page_link("pages/3_Build_Recommender.py", label="Open Build Recommender")

st.markdown(
    """
    **4. Game Recommender**

    The Game Recommender page is where users select a game they like and receive similar
    game suggestions. Users can filter recommendations by platform, genre, price type,
    release year, minimum rating, minimum review count, and number of results. Each result
    includes a match score and shared reasons to make the recommendation easier to explain.
    """
)
st.page_link("pages/4_Game_Recommender.py", label="Open Game Recommender")

st.divider()

st.subheader("Recommendation Method")

st.markdown(
    """
    The main recommendation method is **content-based filtering**. Each game is represented
    using selected textual metadata, such as genre names, Steam tags, categories, and short
    descriptions. These text features are combined into one document per game, then converted
    into numerical vectors using **TF-IDF**. TF-IDF gives higher importance to terms that
    describe a game well while reducing the influence of very common words.

    After the TF-IDF vectors are built, the app compares the selected game with candidate
    games using **cosine similarity**. A higher cosine similarity means the games have more
    similar content descriptions. The recommender also adds genre, tag, and category overlap
    so that recommendations stay close to the selected game's actual taste profile.

    The final match score blends content similarity with optional quality and popularity
    signals. Rating percentage, review count, estimated owners, peak concurrent users, and
    release year are normalized before being used, so features with large raw values do not
    dominate the score unfairly. Popularity is also dampened to reduce the chance that very
    famous games overpower smaller but more relevant games.
    """
)
