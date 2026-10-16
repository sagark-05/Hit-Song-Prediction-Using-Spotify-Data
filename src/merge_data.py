import random
import sys
from pathlib import Path

import pandas as pd

try:
    from .clean_data import normalize_billboard, normalize_spotify
except ImportError:  # pragma: no cover
    from clean_data import normalize_billboard, normalize_spotify

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
BILLBOARD_PATH = DATA_DIR / "billboard.csv"
SPOTIFY_PATH = DATA_DIR / "spotify.csv"
OUTPUT_PATH = DATA_DIR / "final_dataset.csv"
HIT_COUNT = 4000
NON_HIT_COUNT = 6000

FEATURE_COLUMNS = [
    "name",
    "artists",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def load_normalized_datasets():
    billboard = pd.read_csv(BILLBOARD_PATH)
    spotify = pd.read_csv(SPOTIFY_PATH)
    return normalize_billboard(billboard), normalize_spotify(spotify)


def find_billboard_spotify_matches(billboard_df: pd.DataFrame, spotify_df: pd.DataFrame):
    billboard_unique = billboard_df.drop_duplicates(subset=["normalized_key"]).copy()
    spotify_unique = spotify_df.drop_duplicates(subset=["normalized_key"]).copy()

    matched_keys = billboard_unique.loc[
        billboard_unique["normalized_key"].isin(spotify_unique["normalized_key"]),
        "normalized_key",
    ].unique()

    return billboard_unique, spotify_unique, matched_keys


def build_final_dataset(random_seed: int = 42):
    billboard_df, spotify_df = load_normalized_datasets()
    _, spotify_unique, matched_keys = find_billboard_spotify_matches(billboard_df, spotify_df)

    hit_keys = list(matched_keys)
    remaining_keys = spotify_unique.loc[
        ~spotify_unique["normalized_key"].isin(hit_keys),
        "normalized_key",
    ].unique()

    rng = random.Random(random_seed)
    selected_hit_keys = rng.sample(hit_keys, HIT_COUNT)
    selected_nonhit_keys = rng.sample(list(remaining_keys), NON_HIT_COUNT)

    hit_rows = spotify_unique[spotify_unique["normalized_key"].isin(selected_hit_keys)][
        ["name", "artists", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    ].copy()
    hit_rows["hit"] = 1

    nonhit_rows = spotify_unique[spotify_unique["normalized_key"].isin(selected_nonhit_keys)][
        ["name", "artists", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    ].copy()
    nonhit_rows["hit"] = 0

    final_df = pd.concat([hit_rows, nonhit_rows], ignore_index=True)
    final_df = final_df[
        [
            "name",
            "artists",
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "hit",
        ]
    ].copy()

    final_df = final_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    return final_df


def print_verification_report(df: pd.DataFrame):
    total_rows = len(df)
    hit_count = int((df["hit"] == 1).sum())
    non_hit_count = int((df["hit"] == 0).sum())
    duplicate_row_count = int(df.duplicated().sum())
    missing_values = df.isna().sum()

    print("=== FINAL DATASET VERIFICATION REPORT ===")
    print(f"total rows: {total_rows}")
    print(f"hit count: {hit_count}")
    print(f"non-hit count: {non_hit_count}")
    print(f"hit percentage: {hit_count / total_rows * 100:.2f}%")
    print(f"non-hit percentage: {non_hit_count / total_rows * 100:.2f}%")
    print(f"duplicate row count: {duplicate_row_count}")
    print("missing values for every column:")
    print(missing_values.to_string())
    print()

    assert total_rows == HIT_COUNT + NON_HIT_COUNT
    assert hit_count == HIT_COUNT
    assert non_hit_count == NON_HIT_COUNT
    assert duplicate_row_count == 0
    assert missing_values[FEATURE_COLUMNS + ["hit"]].sum() == 0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    final_df = build_final_dataset(random_seed=42)
    print_verification_report(final_df)
    print("=== FIRST 10 ROWS OF final_dataset.csv ===")
    print(final_df.head(10).to_string(index=False))
    print()
    print("Created/modified files:")
    print(f"- {BILLBOARD_PATH}")
    print(f"- {SPOTIFY_PATH}")
    print(f"- {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
