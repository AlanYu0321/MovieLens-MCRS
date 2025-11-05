
"""
data_loader.py — Data loading & preprocessing utilities for MovieLens-MCRS.

Usage:
    from src.data_loader import (
        load_ratings, load_movies, load_tags
    )
"""

from __future__ import annotations
import os
import logging
from typing import Tuple, Optional, List, Dict
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("data_loader")

# ---------- Paths ----------
def _resolve(path: str) -> str:
    return os.path.abspath(path)

# ---------- Loaders ----------
def load_ratings(path: str) -> pd.DataFrame:
    """Load ratings.csv; robust to numeric or string timestamps."""
    path = _resolve(path)
    logger.info(f"Loading ratings from {path}")

    # For colume timestamp, we will try to parse it later as either numeric or string
    dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32"}
    df = pd.read_csv(path, dtype=dtypes, low_memory=False)

    if "timestamp" in df.columns:
        # Transform timestamp to datetime
        ts_numeric = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts_numeric.notna().mean() > 0.95:
            # Timestamp is mostly numeric -> parse as unix epoch seconds
            df["datetime"] = (
                pd.to_datetime(ts_numeric.astype("int64"), unit="s", utc=True)
                  .dt.tz_convert(None)
            )
        else:
            # Parse timestamp as string
            df["datetime"] = (
                pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
                  .dt.tz_convert(None)
            )
    else:
        raise ValueError("ratings.csv has no 'timestamp' column")

    return df

def load_movies(path: str) -> pd.DataFrame:
    """Load movies.csv; keep title and pipe-separated genres string."""
    path = _resolve(path)
    logger.info(f"Loading movies from {path}")
    dtypes = {"movieId": "int32", "title": "string", "genres": "string"}
    return pd.read_csv(path, dtype=dtypes)

def load_tags(path: str) -> pd.DataFrame:
    """Load tags.csv; tags are optional (sparse)."""
    path = _resolve(path)
    logger.info(f"Loading tags from {path}")
    dtypes = {"userId": "int32", "movieId": "int32", "tag": "string", "timestamp": "int64"}
    df = pd.read_csv(path, dtype=dtypes)
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
    return df


# ---------- Merge / Filtering ----------
def merge_ratings_movies(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Inner-join ratings with movie metadata."""
    logger.info("Merging ratings with movies")
    df = ratings.merge(movies, on="movieId", how="inner", validate="many_to_one")
    return df

# ---------- Filtering ----------
def filter_cold_start(
    df: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 50
) -> pd.DataFrame:
    """Remove users/movies with too-few interactions (iterative until stable)."""
    logger.info(f"Filtering cold-start users (<{min_user_ratings}) and movies (<{min_movie_ratings})")
    # Make sure while loop continue.
    prev_shape = (-1, -1)
    while df.shape != prev_shape:
        prev_shape = df.shape
        user_counts = df.groupby("userId", observed=True)["movieId"].count()
        keep_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df["userId"].isin(keep_users)]
        movie_counts = df.groupby("movieId", observed=True)["userId"].count()
        keep_movies = movie_counts[movie_counts >= min_movie_ratings].index
        df = df[df["movieId"].isin(keep_movies)]
    logger.info(f"Remaining: {df['userId'].nunique()} users, {df['movieId'].nunique()} movies, {len(df)} rows")
    return df

# ---------- Features ----------
def encode_genres_multihot(movies: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert pipe-separated genres to multi-hot columns.
    Returns (movies_with_multihot, genre_columns).
    """
    genres = movies["genres"].fillna("")
    unique = set()
    for g in genres:
        unique.update(g.split("|"))
    unique.discard("(no genres listed)")
    unique.discard("")
    genre_cols = sorted(unique)
    out = movies.copy()
    for g in genre_cols:
        out[f"{g}"] = movies["genres"].str.contains(fr"\b{g}\b", regex=True).astype("int8")
    return out, [f"{g}" for g in genre_cols]

# ---------- Splits ----------
def train_valid_test_split_by_time(
    ratings: pd.DataFrame,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    by_user: bool = True,
    timestamp_col: str = "datetime"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split. If by_user=True, split within each user by chronological order.
    """
    assert 0 < valid_ratio < 0.5 and 0 < test_ratio < 0.5 and valid_ratio + test_ratio < 0.9
    if by_user:
        logger.info("Performing per-user chronological split")
        parts = []
        for uid, g in ratings.sort_values(timestamp_col).groupby("userId", sort=False):
            n = len(g)
            n_test = max(1, int(n * test_ratio))
            n_valid = max(1, int(n * valid_ratio))
            test = g.tail(n_test) # Earliest interactions
            valid = g.iloc[-(n_test + n_valid):-n_test] if n - (n_test + n_valid) >= 1 else g.iloc[:0] # Middle interactions
            train = g.iloc[: n - (len(valid) + len(test))] # Latest interactions
            parts.append((train, valid, test))
        train = pd.concat([p[0] for p in parts], ignore_index=True)
        valid = pd.concat([p[1] for p in parts], ignore_index=True)
        test  = pd.concat([p[2] for p in parts], ignore_index=True)
    else:
        logger.info("Performing global chronological split")
        r = ratings.sort_values(timestamp_col)
        n = len(r)
        n_test = int(n * test_ratio)
        n_valid = int(n * valid_ratio)
        test  = r.tail(n_test)
        valid = r.iloc[-(n_test + n_valid):-n_test]
        train = r.iloc[: n - (n_test + n_valid)]
    logger.info(f"Split sizes -> train:{len(train)} valid:{len(valid)} test:{len(test)}")
    return train, valid, test

# ---------- Save ----------
def save_dataframe(df: pd.DataFrame, path: str, index: bool=False) -> None:
    """Save as parquet if path ends with .parquet, else CSV."""
    path = _resolve(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=index)
    else:
        df.to_csv(path, index=index)
    logger.info(f"Saved: {path}")