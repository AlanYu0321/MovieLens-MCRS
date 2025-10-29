"""Neural collaborative filtering (NCF) model and helper utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset


class NCF(nn.Module):
    """Simple MLP-based collaborative filtering model using user/item embeddings."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        hidden: Sequence[int] = (128, 64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        layers: list[nn.Module] = []
        in_dim = emb_dim * 2
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        user_vec = self.user_emb(users)
        item_vec = self.item_emb(items)
        x = torch.cat([user_vec, item_vec], dim=1)
        return self.mlp(x).squeeze(-1)


def build_ncf_model(
    n_users: int,
    n_items: int,
    emb_dim: int = 64,
    hidden: Sequence[int] = (128, 64, 32),
    dropout: float = 0.2,
    device: torch.device | str | None = None,
) -> NCF:
    """Instantiate the NCF model and optionally move it to a device."""

    model = NCF(
        n_users=n_users,
        n_items=n_items,
        emb_dim=emb_dim,
        hidden=hidden,
        dropout=dropout,
    )
    if device is not None:
        model = model.to(device)
    return model


def load_ncf_checkpoint(
    checkpoint_path: str | Path,
    n_users: int,
    n_items: int,
    emb_dim: int = 64,
    hidden: Sequence[int] = (128, 64, 32),
    dropout: float = 0.2,
    device: torch.device | str | None = None,
) -> NCF:
    """Recreate the NCF model architecture and load a saved state dict."""

    checkpoint_path = Path(checkpoint_path)
    map_location = device if device is not None else "cpu"
    state = torch.load(checkpoint_path, map_location=map_location)
    model = build_ncf_model(
        n_users=n_users,
        n_items=n_items,
        emb_dim=emb_dim,
        hidden=hidden,
        dropout=dropout,
        device=device,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def build_id_encoders(
    frames: Iterable[pd.DataFrame],
    user_col: str = "userId",
    item_col: str = "movieId",
) -> Tuple[pd.Series, pd.Series]:
    """Build index encoders for the union of users/items across provided frames."""

    concat_users = pd.concat([df[user_col] for df in frames], ignore_index=True)
    concat_items = pd.concat([df[item_col] for df in frames], ignore_index=True)

    all_users = pd.Index(concat_users.unique())
    all_items = pd.Index(concat_items.unique())

    user2idx = pd.Series(np.arange(len(all_users), dtype=np.int64), index=all_users)
    item2idx = pd.Series(np.arange(len(all_items), dtype=np.int64), index=all_items)
    return user2idx, item2idx


def encode_frame(
    df: pd.DataFrame,
    user2idx: Mapping[int, int],
    item2idx: Mapping[int, int],
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
) -> pd.DataFrame:
    """Add encoded user/item indices to a ratings dataframe."""

    encoded = df[[user_col, item_col, rating_col]].copy()
    encoded["user_idx"] = encoded[user_col].map(user2idx).astype("int64")
    encoded["item_idx"] = encoded[item_col].map(item2idx).astype("int64")
    return encoded[["user_idx", "item_idx", rating_col]].rename(columns={rating_col: "rating"})


def encode_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Encode train/valid/test frames and return the encoded splits with mapping series."""

    user2idx, item2idx = build_id_encoders(
        (train_df, valid_df, test_df),
        user_col=user_col,
        item_col=item_col,
    )
    train_enc = encode_frame(train_df, user2idx, item2idx, user_col, item_col, rating_col)
    valid_enc = encode_frame(valid_df, user2idx, item2idx, user_col, item_col, rating_col)
    test_enc = encode_frame(test_df, user2idx, item2idx, user_col, item_col, rating_col)
    return train_enc, valid_enc, test_enc, user2idx, item2idx


class RatingsDataset(Dataset):
    """PyTorch dataset wrapping encoded user/item interactions."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.ratings[idx]


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str | None = None,
) -> Tuple[float, float, float]:
    """Run a full pass over a dataloader and return (loss, rmse, mae)."""

    is_train = optimizer is not None
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.train() if is_train else model.eval()
    losses: list[float] = []
    true_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []

    with torch.set_grad_enabled(is_train):
        for users, items, ratings in loader:
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)

            preds = model(users, items)
            loss = criterion(preds, ratings)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            true_batches.append(ratings.detach().cpu().numpy())
            pred_batches.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(true_batches)
    y_pred = np.concatenate(pred_batches)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return float(np.mean(losses)), rmse, mae


def build_seen_items(
    train_df: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
) -> dict[int, set[int]]:
    """Create a mapping of users to the set of items they have interacted with."""

    grouped = train_df.groupby(user_col)[item_col].apply(set)
    return {int(user): set(items) for user, items in grouped.items()}


def recommend_topk(
    model: nn.Module,
    user_id: int,
    user2idx: Mapping[int, int],
    item2idx: Mapping[int, int],
    train_seen: Mapping[int, set[int]] | None = None,
    k: int = 10,
    device: torch.device | str | None = None,
) -> list[int]:
    """Recommend top-k unseen items for a raw user id."""

    if user_id not in user2idx:
        raise ValueError(f"Unknown user id {user_id}")

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    seen = train_seen.get(user_id, set()) if train_seen is not None else set()

    candidates = np.array(list(item2idx.keys()))
    if len(seen) > 0:
        mask = ~np.isin(candidates, list(seen))
        candidates = candidates[mask]
    if candidates.size == 0:
        return []

    candidate_indices = torch.tensor(
        [item2idx[item] for item in candidates],
        dtype=torch.long,
        device=device,
    )
    user_idx = int(user2idx[user_id])
    user_tensor = torch.full_like(candidate_indices, fill_value=user_idx)

    with torch.no_grad():
        scores = model(user_tensor, candidate_indices).detach().cpu().numpy()

    topk_idx = np.argsort(-scores)[:k]
    return candidates[topk_idx].tolist()


__all__ = [
    "NCF",
    "build_ncf_model",
    "load_ncf_checkpoint",
    "build_id_encoders",
    "encode_frame",
    "encode_splits",
    "RatingsDataset",
    "run_epoch",
    "build_seen_items",
    "recommend_topk",
]
