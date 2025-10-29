# Streamlit application for exploring the MovieLens collaborative recommender.
# streamlit run dashboard/app.py
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ncf_model import load_ncf_checkpoint

st.set_page_config(
    page_title="MovieLens Recommender Dashboard",
    page_icon="🎬",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_split(name: str) -> pd.DataFrame:
    path = ROOT / "data" / "processed" / f"ratings_{name}.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_all_ratings() -> pd.DataFrame:
    frames = [
        load_split("train")[["userId", "movieId", "rating"]],
        load_split("valid")[["userId", "movieId", "rating"]],
        load_split("test")[["userId", "movieId", "rating"]],
    ]
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["userId", "movieId"])


@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "movies_enriched.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "results.csv")


@st.cache_data(show_spinner=False)
def load_top_movies() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "top_movies.csv")


@st.cache_data(show_spinner=False)
def load_genre_stats() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "genre_stats.csv")


@st.cache_data(show_spinner=False)
def load_active_users() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "active_users.csv")


@lru_cache(maxsize=1)
def build_encoders() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    train = load_split("train")
    valid = load_split("valid")
    test = load_split("test")

    all_users = pd.Index(
        pd.concat([train["userId"], valid["userId"], test["userId"]]).unique()
    )
    all_items = pd.Index(
        pd.concat([train["movieId"], valid["movieId"], test["movieId"]]).unique()
    )

    user2idx = pd.Series(np.arange(len(all_users), dtype=np.int64), index=all_users)
    idx2user = pd.Series(all_users.values, index=user2idx.values)
    item2idx = pd.Series(np.arange(len(all_items), dtype=np.int64), index=all_items)
    idx2item = pd.Series(all_items.values, index=item2idx.values)
    return user2idx, idx2user, item2idx, idx2item


@st.cache_resource(show_spinner=False)
def load_model() -> torch.nn.Module:
    user2idx, _, item2idx, _ = build_encoders()
    model_path = ROOT / "models" / "ncf_best.pth"
    model = load_ncf_checkpoint(
        checkpoint_path=model_path,
        n_users=len(user2idx),
        n_items=len(item2idx),
        device="cpu",
    )
    model.eval()
    return model


def get_user_history(user_id: int, limit: int = 10) -> pd.DataFrame:
    ratings = load_all_ratings()
    movies = load_movies()[["movieId", "title", "genres"]]
    history = ratings[ratings["userId"] == user_id].sort_values(
        "rating", ascending=False
    )
    history = history.merge(movies, on="movieId", how="left")
    return history.head(limit)


def recommend_for_user(user_id: int, top_k: int = 10) -> pd.DataFrame:
    user2idx, _, item2idx, idx2item = build_encoders()
    if user_id not in user2idx.index:
        raise ValueError(f"User {user_id} is not available in the encoded dataset.")

    model = load_model()
    device = next(model.parameters()).device

    user_idx = int(user2idx.loc[user_id])
    items_tensor = torch.arange(len(item2idx), dtype=torch.long, device=device)
    user_tensor = torch.full_like(items_tensor, fill_value=user_idx)

    with torch.no_grad():
        scores = model(user_tensor, items_tensor).detach().cpu().numpy()

    recs = pd.DataFrame(
        {
            "movie_idx": np.arange(len(scores), dtype=np.int64),
            "pred_rating": scores,
        }
    )
    recs["movieId"] = recs["movie_idx"].map(idx2item)

    watched = load_all_ratings()
    watched = set(watched.loc[watched["userId"] == user_id, "movieId"].tolist())
    recs = recs[~recs["movieId"].isin(watched)]

    movies = load_movies()[["movieId", "title", "genres"]]
    recs = recs.merge(movies, on="movieId", how="left")
    recs = recs.sort_values("pred_rating", ascending=False).head(top_k)
    recs["pred_rating"] = recs["pred_rating"].round(3)
    return recs[["movieId", "title", "genres", "pred_rating"]]


def render_overview_tab() -> None:
    st.subheader("Model Evaluation")
    results = load_results()
    comparison = results[results["split"] == "test"].set_index("model")[
        ["rmse", "mae", "precision@10", "recall@10", "ndcg@10"]
    ]
    st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)

    metric_names = {
        "rmse": ("RMSE (↓)", "lower is better"),
        "mae": ("MAE (↓)", "lower is better"),
        "precision@10": ("Precision@10 (↑)", "higher is better"),
        "recall@10": ("Recall@10 (↑)", "higher is better"),
        "ndcg@10": ("nDCG@10 (↑)", "higher is better"),
    }
    display_model = "NCF" if "NCF" in comparison.index else comparison.index[0]
    metric_cols = st.columns(len(metric_names))
    for col, (metric_key, (label, help_text)) in zip(metric_cols, metric_names.items()):
        value = comparison.loc[display_model, metric_key]
        if metric_key in {"rmse", "mae"}:
            best_model = comparison[metric_key].idxmin()
        else:
            best_model = comparison[metric_key].idxmax()
        col.metric(
            label,
            f"{value:.3f}",
            help=f"{help_text}. Best model: {best_model}.",
        )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Most Rated Movies")
        top_movies = (
            load_top_movies().head(10).sort_values("n_ratings", ascending=False)
        )
        st.dataframe(top_movies, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### Genre Popularity")
        genre_stats = load_genre_stats().sort_values("count", ascending=False).head(12)
        st.bar_chart(genre_stats, x="genre", y="count", use_container_width=True)

    st.markdown("#### Power Users")
    st.dataframe(
        load_active_users().head(10),
        hide_index=True,
        use_container_width=True,
    )


def render_recommendations_tab() -> None:
    st.subheader("Personalized Recommendations")
    ratings = load_all_ratings()
    unique_users = sorted(ratings["userId"].unique().tolist())
    selected_user = st.selectbox(
        "Choose a user profile",
        options=unique_users,
        format_func=lambda uid: f"User {uid}",
    )
    top_k = st.slider(
        "Number of movies to recommend", min_value=5, max_value=20, value=10
    )

    try:
        recommendations = recommend_for_user(selected_user, top_k=top_k)
        st.markdown(f"##### Suggested movies for user {selected_user}")
        st.dataframe(recommendations, hide_index=True, use_container_width=True)
    except ValueError as err:
        st.error(str(err))
        return

    st.markdown("##### Recent high ratings")
    history = get_user_history(selected_user, limit=10)
    if history.empty:
        st.info("This user has no prior ratings in the dataset.")
    else:
        st.dataframe(
            history[["movieId", "title", "rating", "genres"]],
            hide_index=True,
            use_container_width=True,
        )


def main() -> None:
    st.title("🎬 MovieLens Collaborative Recommender Dashboard")
    st.caption(
        "Explore model quality metrics, dataset insights, and personalized movie suggestions "
        "powered by the Neural Collaborative Filtering model."
    )
    overview_tab, recommender_tab = st.tabs(["Overview", "Recommendations"])
    with overview_tab:
        render_overview_tab()
    with recommender_tab:
        render_recommendations_tab()


if __name__ == "__main__":
    main()
