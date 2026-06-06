import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Data Generator  (stand-in for MovieLens; swap with real CSV loading easily)
# ──────────────────────────────────────────────────────────────────────────────

class MovieDataGenerator:

    MOVIES = [
        ("The Dark Knight",       2008, ["Action","Crime","Drama"],      ["Christopher Nolan","Christian Bale","Heath Ledger"],    9.0),
        ("Inception",             2010, ["Action","Sci-Fi","Thriller"],   ["Christopher Nolan","Leonardo DiCaprio","Ellen Page"],   8.8),
        ("The Shawshank Redemption",1994,["Drama"],                       ["Frank Darabont","Tim Robbins","Morgan Freeman"],        9.3),
        ("Pulp Fiction",          1994, ["Crime","Drama","Thriller"],     ["Quentin Tarantino","John Travolta","Samuel L. Jackson"],8.9),
        ("The Godfather",         1972, ["Crime","Drama"],                ["Francis Ford Coppola","Marlon Brando","Al Pacino"],     9.2),
        ("Interstellar",          2014, ["Adventure","Drama","Sci-Fi"],   ["Christopher Nolan","Matthew McConaughey","Anne Hathaway"],8.7),
        ("The Matrix",            1999, ["Action","Sci-Fi"],              ["Wachowskis","Keanu Reeves","Laurence Fishburne"],       8.7),
        ("Forrest Gump",          1994, ["Drama","Romance"],              ["Robert Zemeckis","Tom Hanks","Robin Wright"],          8.8),
        ("Fight Club",            1999, ["Drama","Thriller"],             ["David Fincher","Brad Pitt","Edward Norton"],           8.8),
        ("Goodfellas",            1990, ["Biography","Crime","Drama"],    ["Martin Scorsese","Robert De Niro","Ray Liotta"],       8.7),
        ("The Silence of the Lambs",1991,["Crime","Drama","Thriller"],   ["Jonathan Demme","Jodie Foster","Anthony Hopkins"],     8.6),
        ("Schindler's List",      1993, ["Biography","Drama","History"],  ["Steven Spielberg","Liam Neeson","Ralph Fiennes"],      8.9),
        ("The Lord of the Rings: Fellowship",2001,["Adventure","Drama","Fantasy"],["Peter Jackson","Elijah Wood","Ian McKellen"],  8.8),
        ("Star Wars: A New Hope", 1977, ["Action","Adventure","Fantasy"], ["George Lucas","Mark Hamill","Harrison Ford"],         8.6),
        ("Avengers: Endgame",     2019, ["Action","Adventure","Drama"],   ["Russo Brothers","Robert Downey Jr.","Chris Evans"],   8.4),
        ("The Prestige",          2006, ["Drama","Mystery","Sci-Fi"],     ["Christopher Nolan","Christian Bale","Hugh Jackman"],  8.5),
        ("Whiplash",              2014, ["Drama","Music"],                ["Damien Chazelle","Miles Teller","J.K. Simmons"],      8.5),
        ("La La Land",            2016, ["Drama","Music","Romance"],      ["Damien Chazelle","Ryan Gosling","Emma Stone"],        8.0),
        ("Parasite",              2019, ["Comedy","Drama","Thriller"],    ["Bong Joon-ho","Song Kang-ho","Choi Woo-shik"],        8.5),
        ("Joker",                 2019, ["Crime","Drama","Thriller"],     ["Todd Phillips","Joaquin Phoenix","Robert De Niro"],   8.4),
        ("1917",                  2019, ["Drama","War"],                  ["Sam Mendes","George MacKay","Dean-Charles Chapman"],  8.3),
        ("Blade Runner 2049",     2017, ["Action","Drama","Sci-Fi"],      ["Denis Villeneuve","Ryan Gosling","Harrison Ford"],    8.0),
        ("Arrival",               2016, ["Drama","Mystery","Sci-Fi"],     ["Denis Villeneuve","Amy Adams","Jeremy Renner"],      7.9),
        ("The Grand Budapest Hotel",2014,["Adventure","Comedy","Crime"],  ["Wes Anderson","Ralph Fiennes","Tony Revolori"],      8.1),
        ("Mad Max: Fury Road",    2015, ["Action","Adventure","Sci-Fi"],  ["George Miller","Tom Hardy","Charlize Theron"],       8.1),
        ("Spirited Away",         2001, ["Animation","Adventure","Family"],["Hayao Miyazaki","Daveigh Chase","Suzanne Pleshette"],8.6),
        ("Your Name",             2016, ["Animation","Drama","Fantasy"],  ["Makoto Shinkai","Ryunosuke Kamiki","Mone Kamishiraishi"],8.4),
        ("Coco",                  2017, ["Animation","Adventure","Family"],["Lee Unkrich","Anthony Gonzalez","Gael García Bernal"],8.4),
        ("Up",                    2009, ["Animation","Adventure","Drama"],["Pete Docter","Edward Asner","Jordan Nagai"],         8.2),
        ("WALL·E",                2008, ["Animation","Adventure","Family"],["Andrew Stanton","Ben Burtt","Elissa Knight"],       8.4),
    ]

    @classmethod
    def build_movies_df(cls) -> pd.DataFrame:
        rows = []
        for i, (title, year, genres, cast, avg_rating) in enumerate(cls.MOVIES, start=1):
            rows.append({
                "movie_id":   i,
                "title":      title,
                "year":       year,
                "genres":     " ".join(genres),
                "cast":       " ".join(w.replace(" ","_") for w in cast),
                "avg_rating": avg_rating,
                "metadata":   f"{' '.join(genres)} {' '.join(w.replace(' ','_') for w in cast)} {year}",
            })
        return pd.DataFrame(rows)

    @classmethod
    def build_ratings_df(cls, n_users: int = 200, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n_movies = len(cls.MOVIES)
        records = []
        for user_id in range(1, n_users + 1):
            # each user rates 30-70% of movies
            n_rated = rng.integers(int(n_movies * 0.3), int(n_movies * 0.7) + 1)
            movie_ids = rng.choice(range(1, n_movies + 1), size=n_rated, replace=False)
            for mid in movie_ids:
                base = cls.MOVIES[mid - 1][4]          # use avg_rating as base
                noise = rng.normal(0, 0.8)
                rating = float(np.clip(round(base + noise, 1), 1.0, 10.0))
                records.append({"user_id": user_id, "movie_id": int(mid), "rating": rating})
        return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Content-Based Recommender
# ──────────────────────────────────────────────────────────────────────────────

class ContentBasedRecommender:
    """
    Uses TF-IDF on combined genre + cast + year metadata and
    ranks candidates by cosine similarity to the query movie.
    """

    def __init__(self):
        self.vectorizer  = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = None
        self.movies_df    = None

    def fit(self, movies_df: pd.DataFrame):
        self.movies_df    = movies_df.reset_index(drop=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df["metadata"])
        print(f"[ContentBased] TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        return self

    def recommend(self, movie_title: str, n: int = 5) -> pd.DataFrame:
        """Return top-n most similar movies (excluding the query itself)."""
        matches = self.movies_df[self.movies_df["title"].str.lower() == movie_title.lower()]
        if matches.empty:
            raise ValueError(f"Movie '{movie_title}' not found in dataset.")
        idx   = matches.index[0]
        query = self.tfidf_matrix[idx]
        sims  = cosine_similarity(query, self.tfidf_matrix).flatten()
        sims[idx] = -1                                  # exclude self
        top_indices = np.argsort(sims)[::-1][:n]
        result = self.movies_df.iloc[top_indices][["title","year","genres","avg_rating"]].copy()
        result["similarity_score"] = np.round(sims[top_indices], 4)
        return result.reset_index(drop=True)

    def get_feature_importance(self, movie_title: str, top_k: int = 10) -> pd.DataFrame:
        """Which TF-IDF features drove the recommendations?"""
        matches = self.movies_df[self.movies_df["title"].str.lower() == movie_title.lower()]
        idx     = matches.index[0]
        feature_names  = self.vectorizer.get_feature_names_out()
        tfidf_scores   = np.asarray(self.tfidf_matrix[idx].todense()).flatten()
        top_features   = np.argsort(tfidf_scores)[::-1][:top_k]
        return pd.DataFrame({
            "feature": feature_names[top_features],
            "tfidf_weight": np.round(tfidf_scores[top_features], 4),
        })


# ──────────────────────────────────────────────────────────────────────────────
# 2. Collaborative Filtering Recommender (SVD / Matrix Factorisation)
# ──────────────────────────────────────────────────────────────────────────────

class CollaborativeFilteringRecommender:
    """
    Builds a user–item rating matrix and applies Truncated SVD
    (Latent Semantic Analysis in rating space) to find latent factors.
    Predictions fill in the missing ratings for each user.
    """

    def __init__(self, n_components: int = 15):
        self.n_components = n_components
        self.svd          = TruncatedSVD(n_components=n_components, random_state=42)
        self.scaler       = MinMaxScaler()
        self.user_matrix   = None     # (users × components)
        self.item_matrix   = None     # (components × items)
        self.user_ids      = None
        self.movie_ids     = None
        self.movies_df     = None
        self.mean_rating   = None

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.movies_df = movies_df
        # Build user–item pivot (fill NaN with 0 after mean-centering)
        pivot = ratings_df.pivot(index="user_id", columns="movie_id", values="rating")
        self.user_ids  = pivot.index.tolist()
        self.movie_ids = pivot.columns.tolist()
        self.mean_rating = pivot.stack().mean()
        matrix = pivot.fillna(0).values        # (U × M)
        # Decompose: U × Σ × Vt
        U = self.svd.fit_transform(matrix)     # (U × k)
        Vt = self.svd.components_              # (k × M)
        self.user_matrix = U
        self.item_matrix = Vt
        # Reconstruct predicted ratings
        self.predicted = np.dot(U, Vt)         # (U × M)
        print(f"[CollabFilter] SVD components: {self.n_components} | "
              f"Explained variance: {self.svd.explained_variance_ratio_.sum():.2%}")
        return self

    def recommend(self, user_id: int, n: int = 5,
                  already_seen: list[int] | None = None) -> pd.DataFrame:
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not in training data.")
        u_idx  = self.user_ids.index(user_id)
        scores = self.predicted[u_idx]         # predicted rating per movie
        seen   = set(already_seen or [])
        results = []
        for rank, m_idx in enumerate(np.argsort(scores)[::-1]):
            mid = self.movie_ids[m_idx]
            if mid in seen:
                continue
            row = self.movies_df[self.movies_df["movie_id"] == mid]
            if row.empty:
                continue
            results.append({
                "title":            row["title"].values[0],
                "year":             row["year"].values[0],
                "genres":           row["genres"].values[0],
                "avg_rating":       row["avg_rating"].values[0],
                "predicted_rating": round(float(scores[m_idx]), 2),
            })
            if len(results) == n:
                break
        return pd.DataFrame(results)

    def evaluate(self, ratings_df: pd.DataFrame) -> dict:
        """RMSE on held-out 20% of ratings."""
        _, test = train_test_split(ratings_df, test_size=0.2, random_state=42)
        preds, actuals = [], []
        for _, row in test.iterrows():
            if row["user_id"] in self.user_ids and row["movie_id"] in self.movie_ids:
                u_idx = self.user_ids.index(row["user_id"])
                m_idx = self.movie_ids.index(row["movie_id"])
                preds.append(self.predicted[u_idx, m_idx])
                actuals.append(row["rating"])
        rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
        mae  = float(np.mean(np.abs(np.array(actuals) - np.array(preds))))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4), "n_test": len(preds)}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Hybrid Recommender
# ──────────────────────────────────────────────────────────────────────────────

class HybridRecommender:
    """
    Blends Content-Based and Collaborative scores with a configurable weight α.
      hybrid_score = α × content_score + (1 – α) × collab_score
    """

    def __init__(self, alpha: float = 0.5):
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        self.alpha   = alpha
        self.content = ContentBasedRecommender()
        self.collab  = CollaborativeFilteringRecommender()

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.content.fit(movies_df)
        self.collab.fit(ratings_df, movies_df)
        self.movies_df = movies_df
        return self

    def recommend(self, user_id: int, liked_movie: str,
                  n: int = 5) -> pd.DataFrame:
        """
        Uses `liked_movie` for content similarity and `user_id` for personal
        preference, then blends the two ranked lists.
        """
        scaler = MinMaxScaler()

        # --- Content scores ---
        try:
            cb = self.content.recommend(liked_movie, n=len(self.movies_df) - 1)
            cb = cb.rename(columns={"similarity_score": "content_score"})
        except ValueError as e:
            print(f"[Hybrid] Content fallback: {e}")
            cb = pd.DataFrame(columns=["title","content_score"])

        # --- Collaborative scores ---
        try:
            cf = self.collab.recommend(user_id, n=len(self.movies_df))
            cf = cf.rename(columns={"predicted_rating": "collab_score"})
        except ValueError as e:
            print(f"[Hybrid] Collab fallback: {e}")
            cf = pd.DataFrame(columns=["title","collab_score"])

        # --- Merge and normalise ---
        merged = pd.merge(
            self.movies_df[["title","year","genres","avg_rating"]],
            cb[["title","content_score"]],  on="title", how="left"
        )
        merged = pd.merge(merged, cf[["title","collab_score"]], on="title", how="left")
        merged["content_score"].fillna(0, inplace=True)
        merged["collab_score"].fillna(merged["collab_score"].median(), inplace=True)

        if merged["content_score"].max() > 0:
            merged["content_score"] = scaler.fit_transform(
                merged[["content_score"]])
        if merged["collab_score"].max() > 0:
            merged["collab_score"] = scaler.fit_transform(
                merged[["collab_score"]])

        merged["hybrid_score"] = (
            self.alpha * merged["content_score"] +
            (1 - self.alpha) * merged["collab_score"]
        )
        # Exclude the query movie
        merged = merged[merged["title"].str.lower() != liked_movie.lower()]
        top = merged.nlargest(n, "hybrid_score").reset_index(drop=True)
        top["hybrid_score"] = top["hybrid_score"].round(4)
        return top[["title","year","genres","avg_rating",
                    "content_score","collab_score","hybrid_score"]]


# ──────────────────────────────────────────────────────────────────────────────
# Demo / CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🎬  Movie Recommendation System  (ML Demo)")
    print("=" * 60)

    # Build datasets
    movies_df  = MovieDataGenerator.build_movies_df()
    ratings_df = MovieDataGenerator.build_ratings_df(n_users=200)

    print(f"\n📊 Dataset: {len(movies_df)} movies | {len(ratings_df)} ratings "
          f"| {ratings_df['user_id'].nunique()} users\n")

    # ── 1. Content-Based ──────────────────────────────────────────────────────
    print("─" * 60)
    print("1️⃣  CONTENT-BASED FILTERING")
    print("─" * 60)
    cb = ContentBasedRecommender().fit(movies_df)

    query_movie = "Inception"
    recs = cb.recommend(query_movie, n=5)
    print(f"\nTop-5 movies similar to '{query_movie}':\n")
    print(recs.to_string(index=False))

    print(f"\nKey TF-IDF features for '{query_movie}':")
    print(cb.get_feature_importance(query_movie, top_k=8).to_string(index=False))

    # ── 2. Collaborative Filtering ────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("2️⃣  COLLABORATIVE FILTERING (SVD)")
    print("─" * 60)
    cf = CollaborativeFilteringRecommender(n_components=15)
    cf.fit(ratings_df, movies_df)

    user_id = 42
    user_seen = ratings_df[ratings_df["user_id"] == user_id]["movie_id"].tolist()
    cf_recs   = cf.recommend(user_id, n=5, already_seen=user_seen)
    print(f"\nTop-5 personalised picks for User {user_id}:\n")
    print(cf_recs.to_string(index=False))

    metrics = cf.evaluate(ratings_df)
    print(f"\nEvaluation → RMSE: {metrics['rmse']} | MAE: {metrics['mae']} "
          f"(on {metrics['n_test']} held-out ratings)")

    # ── 3. Hybrid ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("3️⃣  HYBRID RECOMMENDER  (α=0.6 content / 0.4 collab)")
    print("─" * 60)
    hybrid = HybridRecommender(alpha=0.6)
    hybrid.fit(movies_df, ratings_df)
    h_recs = hybrid.recommend(user_id=42, liked_movie="The Dark Knight", n=5)
    print(f"\nHybrid top-5 for User 42 who likes 'The Dark Knight':\n")
    print(h_recs.to_string(index=False))

    print("\n✅  All three recommendation strategies completed successfully!")


if __name__ == "__main__":
    main()
