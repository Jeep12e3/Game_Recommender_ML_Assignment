# Data Folder

Download the Kaggle dataset archive and place one of these files here:

```text
data/games_march2025_cleaned.csv
```

or:

```text
data/archive (2).zip
```

If the zip is used, the app reads `games_march2025_cleaned.csv` from inside the archive.

Large CSV/ZIP files are ignored by Git so the repository stays small enough for GitHub.

The app may create a local processed cache such as:

```text
data/processed_games_v2.parquet
```

This cache is also ignored by Git.
