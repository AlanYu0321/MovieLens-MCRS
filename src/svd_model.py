"""svd_model.py — Utilities for training and evaluating Surprise SVD / SVD++ recommenders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple, Type

import numpy as np
import pandas as pd
from surprise import AlgoBase, Dataset, Reader, accuracy
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms import SVD, SVDpp
from surprise.trainset import Trainset


@dataclass
class RatingMetrics:
    """Simple container for regression-style rating metrics."""

    rmse: float
    mae: float


def build_trainset(
    ratings: pd.DataFrame,
    rating_col: str = "rating",
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_scale: Tuple[float, float] | None = None,
) -> Tuple[Trainset, Reader]:
    """
    Convert a dataframe of (user, item, rating) into a Surprise full trainset.
    """

    if rating_scale is None:
        rating_scale = (
            float(ratings[rating_col].min()),
            float(ratings[rating_col].max()),
        )
    reader = Reader(rating_scale=rating_scale)
    dataset = Dataset.load_from_df(
        ratings[[user_col, item_col, rating_col]],
        reader,
    )
    return dataset.build_full_trainset(), reader


def to_prediction_tuples(
    df: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
) -> list[Tuple[int, int, float]]:
    """Helper: dataframe -> list of (user, item, rating) tuples."""

    return list(df[[user_col, item_col, rating_col]].itertuples(index=False, name=None))


def fit_svd(
    trainset: Trainset,
    algo_cls: Type[AlgoBase] = SVD,
    **algo_kwargs,
) -> AlgoBase:
    """
    Train a Surprise algorithm (SVD, SVD++, etc.) on the provided full trainset.
    """

    algo = algo_cls(**algo_kwargs)
    algo.fit(trainset)
    return algo


def predict_pairs(
    algo: AlgoBase,
    pairs: Iterable[Tuple[int, int, float]],
) -> list:
    """
    Generate Surprise predictions for an iterable of (user, item, true_rating).
    """

    return [algo.predict(uid, iid, r_ui) for (uid, iid, r_ui) in pairs]


def rating_metrics(predictions: Sequence) -> RatingMetrics:
    """
    Compute RMSE / MAE given Surprise prediction objects.
    """

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    return RatingMetrics(rmse=rmse, mae=mae)


def grid_search_svd(
    train_df: pd.DataFrame,
    param_grid: Mapping[str, Sequence],
    algo_cls: Type[AlgoBase] = SVD,
    measures: Sequence[str] = ("rmse",),
    cv: int = 3,
    n_jobs: int = -1,
    reader: Reader | None = None,
) -> GridSearchCV:
    """
    Run a Surprise GridSearchCV over the provided dataframe and parameter grid.
    """

    if reader is None:
        reader = Reader(
            rating_scale=(
                float(train_df["rating"].min()),
                float(train_df["rating"].max()),
            )
        )
    data = Dataset.load_from_df(train_df[["userId", "movieId", "rating"]], reader)
    gs = GridSearchCV(
        algo_cls,
        param_grid,
        measures=measures,
        cv=cv,
        n_jobs=n_jobs,
    )
    gs.fit(data)
    return gs


def _user_item_maps(
    train_df: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
) -> Tuple[dict[int, set[int]], np.ndarray]:
    """
    Pre-compute (user -> seen items) mapping and array of all candidate items.
    """

    grouped = train_df.groupby(user_col)[item_col].apply(set)
    user_items = grouped.to_dict()
    all_items = train_df[item_col].unique()
    return user_items, all_items


def recommend_top_k(
    algo: AlgoBase,
    user_id: int,
    train_df: pd.DataFrame,
    k: int = 10,
    user_items_cache: dict[int, set[int]] | None = None,
    all_items: Iterable[int] | None = None,
    item_col: str = "movieId",
) -> list[int]:
    """
    Recommend top-k unseen items for a user using a trained Surprise algorithm.
    """

    if user_items_cache is None or all_items is None:
        user_items_cache, all_items_arr = _user_item_maps(train_df, item_col=item_col)
        all_items = all_items_arr
    seen = user_items_cache.get(user_id, set())
    candidates = [item for item in all_items if item not in seen]
    if not candidates:
        return []
    predictions = [(iid, algo.predict(user_id, iid).est) for iid in candidates]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:k]


def precision_recall_at_k(
    algo: AlgoBase,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    k: int = 10,
    threshold: float = 4.0,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
) -> Tuple[float, float]:
    """
    Compute mean Precision@k and Recall@k on a validation dataframe.
    """

    user_items_cache, all_items = _user_item_maps(train_df, user_col=user_col, item_col=item_col)
    positive_valid = (
        valid_df[valid_df[rating_col] >= threshold]
        .groupby(user_col)[item_col]
        .apply(set)
        .to_dict()
    )

    precisions = []
    recalls = []
    users = sorted(set(train_df[user_col]).intersection(valid_df[user_col].unique()))
    for uid in users:
        positives = positive_valid.get(uid)
        if not positives:
            continue
        recs = recommend_top_k(
            algo,
            user_id=uid,
            train_df=train_df,
            k=k,
            user_items_cache=user_items_cache,
            all_items=all_items,
            item_col=item_col,
        )
        if not recs:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        hit_count = len(set(recs) & positives)
        precisions.append(hit_count / k)
        recalls.append(hit_count / len(positives))

    if not precisions:
        return 0.0, 0.0
    return float(np.mean(precisions)), float(np.mean(recalls))


__all__ = [
    "RatingMetrics",
    "build_trainset",
    "fit_svd",
    "predict_pairs",
    "to_prediction_tuples",
    "rating_metrics",
    "grid_search_svd",
    "recommend_top_k",
    "precision_recall_at_k",
    "SVD",
    "SVDpp",
]
