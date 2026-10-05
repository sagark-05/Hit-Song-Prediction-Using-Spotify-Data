import ast
import re
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_spotify_artists(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            artists = [str(item).strip() for item in parsed if str(item).strip()]
            return " & ".join(artists)
        if isinstance(parsed, tuple):
            artists = [str(item).strip() for item in parsed if str(item).strip()]
            return " & ".join(artists)
    except (ValueError, SyntaxError):
        pass

    cleaned = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"\s+,\s*", " ", cleaned)
    return cleaned.strip()


def normalize_matching_key(title: Any, artist: Any) -> str:
    return f"{normalize_text(title)}|{normalize_text(artist)}"


def normalize_billboard(df: pd.DataFrame) -> pd.DataFrame:
    billboard = df.copy()
    billboard["normalized_title"] = billboard["title"].apply(normalize_text)
    billboard["normalized_artist"] = billboard["performer"].apply(normalize_text)
    billboard["normalized_key"] = billboard.apply(
        lambda row: normalize_matching_key(row["title"], row["performer"]), axis=1
    )
    return billboard


def normalize_spotify(df: pd.DataFrame) -> pd.DataFrame:
    spotify = df.copy()
    spotify["normalized_name"] = spotify["name"].apply(normalize_text)
    spotify["normalized_artist_raw"] = spotify["artists"].apply(parse_spotify_artists)
    spotify["normalized_artist"] = spotify["normalized_artist_raw"].apply(normalize_text)
    spotify["normalized_key"] = spotify.apply(
        lambda row: normalize_matching_key(row["name"], row["artists"]), axis=1
    )
    return spotify
