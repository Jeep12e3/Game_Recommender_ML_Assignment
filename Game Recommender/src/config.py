from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

ENV_DATA_PATH = os.getenv("STEAM_DATA_PATH")
PRIMARY_CSV = DATA_DIR / "games_march2025_cleaned.csv"
PRIMARY_ZIP = DATA_DIR / "archive (2).zip"
LOCAL_DATA_PATH_FILE = DATA_DIR / "local_data_path.txt"
PROCESSED_PARQUET = DATA_DIR / "processed_games_v2.parquet"
CSV_INSIDE_ZIP = "games_march2025_cleaned.csv"

DEFAULT_FEATURES = {
    "genres": True,
    "tags": True,
    "categories": True,
    "short_description": True,
    "developers": False,
    "publishers": False,
}

DEFAULT_WEIGHTS = {
    "content": 0.70,
    "rating": 0.15,
    "popularity": 0.10,
    "recency": 0.05,
}
