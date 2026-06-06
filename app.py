from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from recommender import (
    MovieDataGenerator,
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    HybridRecommender,
)

app = Flask(__name__)
CORS(app)   # allow cross-origin requests (for a separate frontend)

# ── Boot: build data & train models once at startup ──────────────────────────
print("🔄  Loading data and training models…")
movies_df  = MovieDataGenerator.build_movies_df()
ratings_df = MovieDataGenerator.build_ratings_df(n_users=200)

cb_model  = ContentBasedRecommender().fit(movies_df)
cf_model  = CollaborativeFilteringRecommender(n_components=15).fit(ratings_df, movies_df)
hyb_model = HybridRecommender(alpha=0.5).fit(movies_df, ratings_df)
print("✅  Models ready.\n")


# ── Helper ────────────────────────────────────────────────────────────────────
def df_to_json(df):
    return df.where(df.notna(), None).to_dict(orient="records")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "movies": len(movies_df), "users": ratings_df["user_id"].nunique()})


@app.route("/api/movies")
def list_movies():
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 10))
    genre    = request.args.get("genre", "").strip()

    filtered = movies_df.copy()
    if genre:
        filtered = filtered[filtered["genres"].str.contains(genre, case=False)]

    total  = len(filtered)
    start  = (page - 1) * per_page
    subset = filtered.iloc[start: start + per_page]

    return jsonify({
        "page":    page,
        "per_page": per_page,
        "total":   total,
        "movies":  df_to_json(subset[["movie_id","title","year","genres","avg_rating"]]),
    })


@app.route("/api/recommend/content")
def recommend_content():
    title = request.args.get("title", "").strip()
    n     = min(int(request.args.get("n", 5)), 20)
    if not title:
        abort(400, "Query parameter 'title' is required.")
    try:
        recs = cb_model.recommend(title, n=n)
    except ValueError as e:
        abort(404, str(e))
    return jsonify({"query": title, "strategy": "content-based", "recommendations": df_to_json(recs)})


@app.route("/api/recommend/collab")
def recommend_collab():
    user_id = int(request.args.get("user", 0))
    n       = min(int(request.args.get("n", 5)), 20)
    if not user_id:
        abort(400, "Query parameter 'user' (integer) is required.")
    seen = ratings_df[ratings_df["user_id"] == user_id]["movie_id"].tolist()
    try:
        recs = cf_model.recommend(user_id, n=n, already_seen=seen)
    except ValueError as e:
        abort(404, str(e))
    return jsonify({"user_id": user_id, "strategy": "collaborative-svd", "recommendations": df_to_json(recs)})


@app.route("/api/recommend/hybrid")
def recommend_hybrid():
    user_id = int(request.args.get("user", 0))
    title   = request.args.get("title", "").strip()
    alpha   = float(request.args.get("alpha", 0.5))
    n       = min(int(request.args.get("n", 5)), 20)

    if not user_id or not title:
        abort(400, "'user' and 'title' are required.")

    hyb_model.alpha = max(0.0, min(1.0, alpha))
    try:
        recs = hyb_model.recommend(user_id, liked_movie=title, n=n)
    except ValueError as e:
        abort(404, str(e))
    return jsonify({
        "user_id": user_id, "liked_movie": title, "alpha": alpha,
        "strategy": "hybrid", "recommendations": df_to_json(recs),
    })


@app.route("/api/user/<int:user_id>/history")
def user_history(user_id):
    history = ratings_df[ratings_df["user_id"] == user_id].copy()
    if history.empty:
        abort(404, f"User {user_id} not found.")
    merged = history.merge(
        movies_df[["movie_id","title","year","genres"]], on="movie_id"
    ).sort_values("rating", ascending=False)
    return jsonify({"user_id": user_id, "rated_movies": df_to_json(merged[["title","year","genres","rating"]])})


@app.route("/api/movies/search")
def search_movies():
    query = request.args.get("q", "").strip()
    if not query:
        abort(400, "Query parameter 'q' is required.")
    mask   = movies_df["title"].str.contains(query, case=False)
    result = movies_df[mask][["movie_id","title","year","genres","avg_rating"]]
    return jsonify({"query": query, "results": df_to_json(result)})


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
