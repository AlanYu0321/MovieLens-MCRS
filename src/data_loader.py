
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


