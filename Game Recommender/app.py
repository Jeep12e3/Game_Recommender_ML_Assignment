import streamlit as st


st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="🎮",
    layout="wide",
)

pages = [
    st.Page("Home.py", title="Home"),
    st.Page("pages/1_Raw_Data_EDA.py", title="Exploratory Data Analysis"),
    st.Page("pages/2_Preprocessing.py", title="Preprocessing"),
    st.Page("pages/3_Build_Recommender.py", title="Build Recommender"),
    st.Page("pages/4_Game_Recommender.py", title="Game Recommender"),
]

st.navigation(pages).run()
