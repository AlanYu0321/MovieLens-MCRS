"""autoencoder_model.py — Denoising autoencoder utilities for collaborative filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class AutoEncoderDataset(Dataset):
    """Lightweight dataset feeding whole-user rating vectors and masks."""

    def __init__(self, ratings: np.ndarray, mask: np.ndarray) -> None:
        assert ratings.shape == mask.shape, "ratings/mask must share shape"
        self.ratings = torch.from_numpy(ratings.astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(np.float32))

    def __len__(self) -> int:
        return self.ratings.shape[0]

    def __getitem__(self, idx: int):
        return self.ratings[idx], self.mask[idx]


class DenoisingAutoEncoder(nn.Module):
    """Symmetric MLP autoencoder that reconstructs item-rating vectors."""

    def __init__(
        self,
        n_items: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        dims = [n_items, *hidden_dims]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))

        decoder_layers: list[nn.Module] = []
        rev_dims = [*hidden_dims[::-1], n_items]
        for i, (in_dim, out_dim) in enumerate(zip(rev_dims[:-1], rev_dims[1:])):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < len(rev_dims) - 2:
                decoder_layers.append(nn.ReLU())
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self._reset()

    def _reset(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


def build_autoencoder(
    n_items: int,
    hidden_dims: Sequence[int] = (512, 256, 128),
    dropout: float = 0.25,
    device: torch.device | str | None = None,
) -> DenoisingAutoEncoder:
    model = DenoisingAutoEncoder(n_items=n_items, hidden_dims=hidden_dims, dropout=dropout)
    if device is not None:
        model = model.to(device)
    return model


def load_autoencoder_checkpoint(
    checkpoint_path: str | Path,
    n_items: int,
    hidden_dims: Sequence[int] = (512, 256, 128),
    dropout: float = 0.25,
    device: torch.device | str | None = None,
) -> tuple[DenoisingAutoEncoder, dict]:
    """Load the denoising autoencoder along with its stored metadata."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device if device is not None else "cpu")
    model = build_autoencoder(
        n_items=n_items,
        hidden_dims=hidden_dims,
        dropout=dropout,
        device=device,
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def encode_dense_matrix(
    df: pd.DataFrame,
    user2idx: Mapping[int, int] | None = None,
    item2idx: Mapping[int, int] | None = None,
    user_col: str = 'userId',
    item_col: str = 'movieId',
    rating_col: str = 'rating',
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Turn a ratings dataframe into dense (users × items) matrix and mask."""

    if user2idx is None:
        users = pd.Index(df[user_col].unique())
        user2idx = pd.Series(np.arange(len(users), dtype=np.int64), index=users)
    else:
        user2idx = pd.Series(user2idx)

    if item2idx is None:
        items = pd.Index(df[item_col].unique())
        item2idx = pd.Series(np.arange(len(items), dtype=np.int64), index=items)
    else:
        item2idx = pd.Series(item2idx)

    matrix = np.zeros((len(user2idx), len(item2idx)), dtype=np.float32)
    mask = np.zeros_like(matrix, dtype=np.float32)

    u_idx = user2idx.loc[df[user_col]].to_numpy()
    i_idx = item2idx.loc[df[item_col]].to_numpy()
    ratings = df[rating_col].to_numpy(dtype=np.float32)

    matrix[u_idx, i_idx] = ratings
    mask[u_idx, i_idx] = 1.0

    return matrix, mask, user2idx, item2idx


def encode_dense_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_col: str = 'userId',
    item_col: str = 'movieId',
    rating_col: str = 'rating',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Encode train/valid/test dataframes into dense matrices sharing the same indices."""

    union_df = pd.concat([train_df[[user_col, item_col, rating_col]],
                          valid_df[[user_col, item_col, rating_col]],
                          test_df[[user_col, item_col, rating_col]]], ignore_index=True)
    _, _, user2idx, item2idx = encode_dense_matrix(union_df, user_col=user_col, item_col=item_col, rating_col=rating_col)

    train_mat, train_mask, _, _ = encode_dense_matrix(train_df, user2idx=user2idx, item2idx=item2idx, user_col=user_col, item_col=item_col, rating_col=rating_col)
    valid_mat, valid_mask, _, _ = encode_dense_matrix(valid_df, user2idx=user2idx, item2idx=item2idx, user_col=user_col, item_col=item_col, rating_col=rating_col)
    test_mat, test_mask, _, _ = encode_dense_matrix(test_df, user2idx=user2idx, item2idx=item2idx, user_col=user_col, item_col=item_col, rating_col=rating_col)

    return train_mat, train_mask, valid_mat, valid_mask, test_mat, test_mask, user2idx, item2idx


@dataclass
class EpochMetrics:
    loss: float
    rmse: float
    mae: float


def run_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str | None = None,
    corruption: float = 0.2,
    min_rating: float = 0.5,
    max_rating: float = 5.0,
) -> EpochMetrics:
    """Single epoch pass returning loss, RMSE, and MAE on observed entries."""

    training = optimizer is not None
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.train() if training else model.eval()

    total_sqerr = 0.0
    total_abserr = 0.0
    total_count = 0.0
    losses: list[float] = []

    for ratings, mask in loader:
        ratings = ratings.to(device)
        mask = mask.to(device)

        noisy = ratings
        if training and corruption > 0:
            noise_mask = torch.rand_like(ratings) > corruption
            noisy = ratings * noise_mask

        recon = model(noisy).clamp(min=min_rating, max=max_rating)

        diff = (recon - ratings) * mask
        sqerr = torch.sum(diff.pow(2))
        abserr = torch.sum(diff.abs())
        count = torch.sum(mask).clamp_min(1.0)

        loss = sqerr / count

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        total_sqerr += sqerr.item()
        total_abserr += abserr.item()
        total_count += count.item()

    mse = total_sqerr / total_count
    rmse = float(np.sqrt(mse))
    mae = float(total_abserr / total_count)
    return EpochMetrics(loss=float(np.mean(losses)), rmse=rmse, mae=mae)


def build_seen_items(
    ratings: pd.DataFrame,
    user_col: str = 'userId',
    item_col: str = 'movieId',
) -> dict[int, set[int]]:
    """Map each user to the set of items they have interacted with."""

    grouped = ratings.groupby(user_col)[item_col].apply(set)
    return {int(user): set(items) for user, items in grouped.items()}


def recommend_topk(
    model: nn.Module,
    user_id: int,
    train_matrix: np.ndarray,
    user2idx: Mapping[int, int],
    item2idx: Mapping[int, int],
    train_seen: Mapping[int, set[int]] | None = None,
    k: int = 10,
    device: torch.device | str | None = None,
    min_rating: float = 0.5,
    max_rating: float = 5.0,
) -> pd.DataFrame:
    """Recommend top-k items for a user based on reconstructed ratings."""

    user_series = pd.Series(user2idx)
    item_series = pd.Series(item2idx)
    if user_id not in user_series.index:
        raise ValueError(f"Unknown userId {user_id}")

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    user_idx = int(user_series.loc[user_id])
    user_vector = torch.from_numpy(train_matrix[user_idx]).unsqueeze(0).to(device)

    with torch.no_grad():
        recon_raw = model(user_vector).squeeze(0).cpu().numpy()
        recon_clipped = np.clip(recon_raw, min_rating, max_rating)

    seen = train_seen.get(user_id, set()) if train_seen is not None else set()
    candidates = np.array(item_series.index.tolist())
    if seen:
        mask = ~np.isin(candidates, list(seen))
        candidates = candidates[mask]

    if candidates.size == 0:
        return pd.DataFrame(columns=["movieId", "pred_rating"])

    raw_scores = recon_raw[item_series.loc[candidates].to_numpy()]
    clipped_scores = recon_clipped[item_series.loc[candidates].to_numpy()]

    order = np.argsort(-raw_scores)[:k]

    df = pd.DataFrame(
        {
            "movieId": candidates[order],
            "pred_rating": clipped_scores[order],
            "raw_score": raw_scores[order],
        }
    )
    df["pred_rating"] = df["pred_rating"].round(3)
    df["raw_score"] = df["raw_score"].round(3)
    return df


__all__ = [
    'AutoEncoderDataset',
    'DenoisingAutoEncoder',
    'build_autoencoder',
    'load_autoencoder_checkpoint',
    'encode_dense_matrix',
    'encode_dense_splits',
    'EpochMetrics',
    'run_epoch',
    'build_seen_items',
    'recommend_topk',
]
