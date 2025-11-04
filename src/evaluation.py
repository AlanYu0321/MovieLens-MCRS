"""Evaluation helpers for rating prediction and ranking metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ncf_model import recommend_topk as _recommend_topk_ncf
from autoencoder_model import recommend_topk as _recommend_topk_autoencoder_df


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float


def predict_ncf(
    model: torch.nn.Module,
    interactions: pd.DataFrame,
    *,
    user2idx: Mapping[int, int],
    item2idx: Mapping[int, int],
    batch_size: int = 4096,
    device: torch.device | str | None = None,
    clamp: tuple[float, float] | None = (0.5, 5.0),
) -> np.ndarray:
    """Predict ratings with an NCF-style model for the given interactions."""

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    user_series = pd.Series(user2idx)
    item_series = pd.Series(item2idx)
    users = torch.from_numpy(user_series.loc[interactions['userId']].to_numpy())
    items = torch.from_numpy(item_series.loc[interactions['movieId']].to_numpy())

    dataset = TensorDataset(users, items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_u, batch_i in loader:
            batch_u = batch_u.to(device)
            batch_i = batch_i.to(device)
            scores = model(batch_u, batch_i)
            if clamp is not None:
                scores = scores.clamp(*clamp)
            preds.append(scores.cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty(0, dtype=np.float32)


def predict_autoencoder(
    model: torch.nn.Module,
    interactions: pd.DataFrame,
    *,
    train_matrix: np.ndarray,
    user2idx: Mapping[int, int],
    item2idx: Mapping[int, int],
    batch_size: int = 2048,
    device: torch.device | str | None = None,
    clamp: tuple[float, float] | None = (0.5, 5.0),
) -> np.ndarray:
    """Predict ratings using a denoising autoencoder given a dense train matrix."""

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    user_series = pd.Series(user2idx)
    item_series = pd.Series(item2idx)
    user_indices = user_series.loc[interactions['userId']].to_numpy()
    item_indices = item_series.loc[interactions['movieId']].to_numpy()

    dataset = TensorDataset(
        torch.from_numpy(user_indices.astype(np.int64)),
        torch.from_numpy(item_indices.astype(np.int64)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_users, batch_items in loader:
            rows = torch.from_numpy(train_matrix[batch_users.numpy()]).to(device)
            recon = model(rows)
            if clamp is not None:
                recon = recon.clamp(*clamp)
            scores = recon.gather(1, batch_items.to(device).unsqueeze(1)).squeeze(1)
            preds.append(scores.cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty(0, dtype=np.float32)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Return RMSE and MAE for the predicted ratings."""

    y_true = y_true.astype(np.float32)
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    mae = float(np.mean(np.abs(diff)))
    return RegressionMetrics(rmse=rmse, mae=mae)


def summarize_ranking(
    model_name: str,
    split_name: str,
    truth: Mapping[int, Sequence[int] | set[int]],
    recommend_fn: Callable[[int, int], Sequence[int]],
    *,
    k: int = 10,
) -> dict[str, object]:
    """Aggregate precision/recall/nDCG@k for a recommender over the provided truth map."""

    rows: list[dict[str, float]] = []
    for user_id, positives in truth.items():
        items = set(positives)
        if not items:
            continue
        try:
            recs = recommend_fn(user_id, k=k)
        except ValueError:
            continue
        if not recs:
            continue
        hits = [1 if item in items else 0 for item in recs]
        precision = sum(hits) / k
        recall = sum(hits) / len(items)
        dcg = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(len(items), k)
        idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1)) if ideal_hits else 0.0
        ndcg = dcg / idcg if idcg > 0 else 0.0
        rows.append({'precision': precision, 'recall': recall, 'ndcg': ndcg})

    if not rows:
        return {
            'model': model_name,
            'split': split_name,
            'users_evaluated': 0,
            f'precision@{k}': float('nan'),
            f'recall@{k}': float('nan'),
            f'ndcg@{k}': float('nan'),
        }

    stats = pd.DataFrame(rows)
    return {
        'model': model_name,
        'split': split_name,
        'users_evaluated': len(stats),
        f'precision@{k}': float(stats['precision'].mean()),
        f'recall@{k}': float(stats['recall'].mean()),
        f'ndcg@{k}': float(stats['ndcg'].mean()),
    }


def recommend_topk_ncf_ids(
    model: torch.nn.Module,
    user_id: int,
    *,
    user2idx: Mapping[int, int],
    item2idx: Mapping[int, int],
    train_seen: Mapping[int, set[int]] | None = None,
    k: int = 10,
    device: torch.device | str | None = None,
) -> list[int]:
    """Convenience wrapper returning NCF top-k recommendations as raw ids."""

    return _recommend_topk_ncf(
        model,
        user_id,
        user2idx,
        item2idx,
        train_seen=train_seen,
        k=k,
        device=device,
    )


def recommend_topk_autoencoder_ids(
    model: torch.nn.Module,
    user_id: int,
    *,
    train_matrix: np.ndarray,
    user2idx: Mapping[int, int],
    item2idx: Mapping[int, int],
    train_seen: Mapping[int, set[int]] | None = None,
    k: int = 10,
    device: torch.device | str | None = None,
) -> list[int]:
    """Convenience wrapper returning AutoEncoder top-k recommendations as raw ids."""

    df = _recommend_topk_autoencoder_df(
        model,
        user_id,
        train_matrix=train_matrix,
        user2idx=user2idx,
        item2idx=item2idx,
        train_seen=train_seen,
        k=k,
        device=device,
    )
    if df.empty:
        return []
    return df['movieId'].astype(int).tolist()


__all__ = [
    'RegressionMetrics',
    'predict_ncf',
    'predict_autoencoder',
    'compute_regression_metrics',
    'summarize_ranking',
    'recommend_topk_ncf_ids',
    'recommend_topk_autoencoder_ids',
]
