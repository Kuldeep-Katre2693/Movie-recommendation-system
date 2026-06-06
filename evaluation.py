import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# ──────────────────────────────────────────────────────────────────────────────
# Rating-Prediction Metrics
# ──────────────────────────────────────────────────────────────────────────────

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Ranking Metrics (Precision@K, Recall@K, NDCG@K)
# ──────────────────────────────────────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Fraction of top-K recommendations that are actually relevant.
    A movie is "relevant" if the user rated it ≥ relevance_threshold.
    """
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of all relevant items captured in the top-K."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: list, relevant: set, k: int) -> float:
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalised Discounted Cumulative Gain @K."""
    def dcg(items):
        return sum(
            (1.0 / np.log2(rank + 2))
            for rank, item in enumerate(items[:k])
            if item in relevant
        )
    ideal_hits = min(k, len(relevant))
    ideal_dcg  = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg(recommended) / ideal_dcg if ideal_dcg > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Catalogue Coverage & Diversity
# ──────────────────────────────────────────────────────────────────────────────

def catalogue_coverage(all_recommendations: list[list], total_items: int) -> float:
    """What fraction of the full catalogue appears in at least one recommendation list?"""
    recommended_set = set(item for recs in all_recommendations for item in recs)
    return len(recommended_set) / total_items if total_items > 0 else 0.0


def intra_list_diversity(rec_list: list, item_features: dict) -> float:
    """
    Average pairwise distance between recommended items based on genre vectors.
    Higher = more diverse recommendations.
    item_features: {movie_id: genre_vector (np.array)}
    """
    if len(rec_list) < 2:
        return 0.0
    distances = []
    for i in range(len(rec_list)):
        for j in range(i + 1, len(rec_list)):
            f1 = item_features.get(rec_list[i])
            f2 = item_features.get(rec_list[j])
            if f1 is not None and f2 is not None:
                sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-9)
                distances.append(1 - sim)
    return float(np.mean(distances)) if distances else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# K-Fold Cross-Validation for Collaborative Filtering
# ──────────────────────────────────────────────────────────────────────────────

def cross_validate_cf(ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                      n_splits: int = 5, n_components: int = 15) -> dict:
    """
    Run K-fold CV on the collaborative filtering model.
    Returns mean and std of RMSE / MAE across folds.
    """
    from recommender import CollaborativeFilteringRecommender

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores, mae_scores = [], []

    data = ratings_df.values                          # numpy array for indexing
    for fold, (train_idx, test_idx) in enumerate(kf.split(data), start=1):
        train_df = pd.DataFrame(data[train_idx], columns=ratings_df.columns)
        test_df  = pd.DataFrame(data[test_idx],  columns=ratings_df.columns)

        model = CollaborativeFilteringRecommender(n_components=n_components)
        model.fit(train_df, movies_df)

        preds, actuals = [], []
        for _, row in test_df.iterrows():
            uid, mid, true_r = int(row["user_id"]), int(row["movie_id"]), float(row["rating"])
            if uid in model.user_ids and mid in model.movie_ids:
                u_idx = model.user_ids.index(uid)
                m_idx = model.movie_ids.index(mid)
                preds.append(model.predicted[u_idx, m_idx])
                actuals.append(true_r)

        if preds:
            fold_rmse = rmse(np.array(actuals), np.array(preds))
            fold_mae  = mae(np.array(actuals), np.array(preds))
            rmse_scores.append(fold_rmse)
            mae_scores.append(fold_mae)
            print(f"  Fold {fold}/{n_splits} → RMSE: {fold_rmse:.4f} | MAE: {fold_mae:.4f}")

    return {
        "mean_rmse": round(float(np.mean(rmse_scores)), 4),
        "std_rmse":  round(float(np.std(rmse_scores)),  4),
        "mean_mae":  round(float(np.mean(mae_scores)),  4),
        "std_mae":   round(float(np.std(mae_scores)),   4),
        "n_folds":   n_splits,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full Evaluation Report
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_all_models(movies_df: pd.DataFrame,
                        ratings_df: pd.DataFrame,
                        k: int = 5) -> pd.DataFrame:
    """
    Compare Content-Based, Collaborative, and Hybrid models side-by-side
    using Precision@K, Recall@K, NDCG@K, and RMSE.
    """
    from recommender import (ContentBasedRecommender,
                             CollaborativeFilteringRecommender,
                             HybridRecommender)

    RELEVANCE_THRESHOLD = 7.5   # Ratings ≥ 7.5 are "relevant"

    # Build relevance sets per user
    relevant_map = (
        ratings_df[ratings_df["rating"] >= RELEVANCE_THRESHOLD]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    # Train models
    cb  = ContentBasedRecommender().fit(movies_df)
    cf  = CollaborativeFilteringRecommender().fit(ratings_df, movies_df)
    hyb = HybridRecommender(alpha=0.5)
    hyb.fit(movies_df, ratings_df)

    p_scores  = {"content": [], "collab": [], "hybrid": []}
    r_scores  = {"content": [], "collab": [], "hybrid": []}
    nd_scores = {"content": [], "collab": [], "hybrid": []}

    sample_users = list(relevant_map.keys())[:50]    # use first 50 users

    for uid in sample_users:
        relevant = relevant_map[uid]
        seen     = ratings_df[ratings_df["user_id"] == uid]["movie_id"].tolist()

        # Content-Based: recommend based on highest-rated movie for this user
        liked_mid = (
            ratings_df[(ratings_df["user_id"] == uid) & (ratings_df["rating"] >= 8.0)]
            ["movie_id"].tolist()
        )
        if liked_mid:
            title = movies_df[movies_df["movie_id"] == liked_mid[0]]["title"].values[0]
            try:
                cb_recs = cb.recommend(title, n=k)
                cb_ids  = movies_df[movies_df["title"].isin(cb_recs["title"])]["movie_id"].tolist()
                p_scores["content"].append(precision_at_k(cb_ids, relevant, k))
                r_scores["content"].append(recall_at_k(cb_ids, relevant, k))
                nd_scores["content"].append(ndcg_at_k(cb_ids, relevant, k))
            except Exception:
                pass

        # Collaborative
        try:
            cf_recs = cf.recommend(uid, n=k, already_seen=seen)
            cf_ids  = movies_df[movies_df["title"].isin(cf_recs["title"])]["movie_id"].tolist()
            p_scores["collab"].append(precision_at_k(cf_ids, relevant, k))
            r_scores["collab"].append(recall_at_k(cf_ids, relevant, k))
            nd_scores["collab"].append(ndcg_at_k(cf_ids, relevant, k))
        except Exception:
            pass

        # Hybrid
        if liked_mid:
            try:
                h_recs = hyb.recommend(uid, liked_movie=title, n=k)
                h_ids  = movies_df[movies_df["title"].isin(h_recs["title"])]["movie_id"].tolist()
                p_scores["hybrid"].append(precision_at_k(h_ids, relevant, k))
                r_scores["hybrid"].append(recall_at_k(h_ids, relevant, k))
                nd_scores["hybrid"].append(ndcg_at_k(h_ids, relevant, k))
            except Exception:
                pass

    rows = []
    for name in ["content", "collab", "hybrid"]:
        rows.append({
            "Model":        {"content": "Content-Based", "collab": "Collaborative (SVD)",
                             "hybrid": "Hybrid"}[name],
            f"Precision@{k}": round(np.mean(p_scores[name]),  4) if p_scores[name]  else "-",
            f"Recall@{k}":    round(np.mean(r_scores[name]),  4) if r_scores[name]  else "-",
            f"NDCG@{k}":      round(np.mean(nd_scores[name]), 4) if nd_scores[name] else "-",
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Run standalone
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from recommender import MovieDataGenerator

    movies_df  = MovieDataGenerator.build_movies_df()
    ratings_df = MovieDataGenerator.build_ratings_df(n_users=200)

    print("\n📐  5-Fold Cross-Validation (Collaborative Filtering)\n")
    cv_results = cross_validate_cf(ratings_df, movies_df, n_splits=5)
    print(f"\n  Mean RMSE : {cv_results['mean_rmse']} ± {cv_results['std_rmse']}")
    print(f"  Mean MAE  : {cv_results['mean_mae']} ± {cv_results['std_mae']}")

    print("\n\n📊  Model Comparison Report (Precision / Recall / NDCG @ K=5)\n")
    report = evaluate_all_models(movies_df, ratings_df, k=5)
    print(report.to_string(index=False))
