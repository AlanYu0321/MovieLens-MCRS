# Streamlit application for exploring the MovieLens collaborative recommender.
# streamlit run dashboard/app.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from surprise import dump

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.autoencoder_model import (
    encode_dense_splits,
    load_autoencoder_checkpoint,
    recommend_topk as recommend_topk_autoencoder,
)
from src.ncf_model import (
    load_ncf_checkpoint,
    recommend_topk as recommend_topk_ncf,
)
from src.svd_model import recommend_top_k as recommend_topk_svd

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


def get_user_history(user_id: int, limit: int = 10) -> pd.DataFrame:
    ratings = load_all_ratings()
    movies = load_movies()[["movieId", "title", "genres"]]
    history = ratings[ratings["userId"] == user_id].sort_values(
        "rating", ascending=False
    )
    history = history.merge(movies, on="movieId", how="left")
    return history.head(limit)


@st.cache_resource(show_spinner=False)
def load_recommender_interfaces() -> dict[str, callable]:
    train = load_split("train")
    valid = load_split("valid")
    test = load_split("test")
    movies = load_movies()[["movieId", "title", "genres"]]

    (
        train_matrix,
        _,
        _,
        _,
        _,
        _,
        user2idx,
        item2idx,
    ) = encode_dense_splits(train, valid, test)

    user2idx = pd.Series(user2idx)
    item2idx = pd.Series(item2idx)

    train_seen = train.groupby("userId")["movieId"].apply(set).to_dict()
    all_items = train["movieId"].unique()
    empty_df = pd.DataFrame(columns=["movieId", "title", "genres", "pred_rating"])

    def format_recommendations(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return empty_df.copy()
        merged = df.merge(movies, on="movieId", how="left")
        if "pred_rating" in merged.columns:
            merged["pred_rating"] = merged["pred_rating"].round(3)
        else:
            merged["pred_rating"] = np.nan
        return merged[["movieId", "title", "genres", "pred_rating"]]

    ncf_model = load_ncf_checkpoint(
        checkpoint_path=ROOT / "models" / "ncf_best.pth",
        n_users=len(user2idx),
        n_items=len(item2idx),
        device="cpu",
    )

    def recommend_ncf(user_id: int, top_k: int = 10) -> pd.DataFrame:
        if user_id not in user2idx.index:
            raise ValueError(f"User {user_id} is not available in the dataset.")
        recs = recommend_topk_ncf(
            ncf_model,
            user_id,
            user2idx=user2idx,
            item2idx=item2idx,
            train_seen=train_seen,
            k=top_k,
            device="cpu",
        )
        if isinstance(recs, list):
            recs = pd.DataFrame(recs, columns=["movieId", "pred_rating"])
        return format_recommendations(recs)

    ae_model, _ = load_autoencoder_checkpoint(
        ROOT / "models" / "autoencoder_best.pth",
        n_items=len(item2idx),
        device="cpu",
    )

    def recommend_autoencoder(user_id: int, top_k: int = 10) -> pd.DataFrame:
        if user_id not in user2idx.index:
            raise ValueError(f"User {user_id} is not available in the dataset.")
        recs = recommend_topk_autoencoder(
            ae_model,
            user_id,
            train_matrix=train_matrix,
            user2idx=user2idx,
            item2idx=item2idx,
            train_seen=train_seen,
            k=top_k,
            device="cpu",
        )
        if isinstance(recs, list):
            recs = pd.DataFrame(recs, columns=["movieId", "pred_rating"])
        return format_recommendations(recs)

    svd_path = ROOT / "models" / "svd_baseline.dump"
    svd_algo = None
    if svd_path.exists():
        _, svd_algo = dump.load(str(svd_path))

    def recommend_svd(user_id: int, top_k: int = 10) -> pd.DataFrame:
        if svd_algo is None:
            return empty_df.copy()
        recs = recommend_topk_svd(
            svd_algo,
            user_id,
            train_df=train,
            k=top_k,
            user_items_cache=train_seen,
            all_items=all_items,
        )
        if not recs:
            return empty_df.copy()
        df = pd.DataFrame(recs, columns=["movieId", "pred_rating"])
        return format_recommendations(df)

    return {
        "SVD": recommend_svd,
        "NCF": recommend_ncf,
        "AutoEncoder": recommend_autoencoder,
    }


def recommend_for_user(user_id: int, model_name: str, top_k: int = 10) -> pd.DataFrame:
    recommenders = load_recommender_interfaces()
    if model_name not in recommenders:
        raise ValueError(f"Unknown model: {model_name}")
    return recommenders[model_name](user_id, top_k)


def render_overview_tab() -> None:
    st.subheader("Model Evaluation")
    results = load_results()
    split_options = sorted(results["split"].unique())
    default_index = split_options.index("test") if "test" in split_options else 0
    selected_split = st.selectbox("Evaluation split", split_options, index=default_index)
    comparison = results[results["split"] == selected_split].set_index("model")[
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
        is_lower_better = metric_key in {"rmse", "mae"}
        if is_lower_better:
            best_model = comparison[metric_key].idxmin()
            best_value = comparison.loc[best_model, metric_key]
        else:
            best_model = comparison[metric_key].idxmax()
            best_value = comparison.loc[best_model, metric_key]
        col.metric(
            label,
            f"{best_value:.3f}",
            help=f"{help_text}. Best model on {selected_split}: {best_model}.",
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
    recommenders = load_recommender_interfaces()
    model_names = list(recommenders.keys())
    default_model_index = model_names.index("NCF") if "NCF" in model_names else 0

    col1, col2, col3 = st.columns([2, 1, 1])
    selected_user = col1.selectbox(
        "Choose a user profile",
        options=unique_users,
        format_func=lambda uid: f"User {uid}",
    )
    top_k = col2.slider(
        "Number of movies to recommend", min_value=5, max_value=20, value=10
    )
    model_choice = col3.selectbox(
        "Recommender",
        options=model_names,
        index=default_model_index,
    )

    try:
        recommendations = recommend_for_user(selected_user, model_choice, top_k=top_k)
        st.markdown(
            f"##### Suggested movies for user {selected_user} via {model_choice}"
        )
        if recommendations.empty:
            st.info("No recommendations available for this user/model combination.")
        else:
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
        "powered by SVD, Neural Collaborative Filtering, and AutoEncoder recommenders."
    )
    overview_tab, recommender_tab = st.tabs(["Overview", "Recommendations"])
    with overview_tab:
        render_overview_tab()
    with recommender_tab:
        render_recommendations_tab()


if __name__ == "__main__":
    main()
