import os

from sklearn.model_selection import train_test_split

from src.cleaner import DataCleaner
from src.config import ConfigLoader
from src.evaluation import Evaluator
from src.loader import DataLoader
from src.recommenders.baseline import BaselineRecommender
from src.recommenders.collaborative import CollaborativeRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import EnsembleRecommender, MetaEnsembleRecommender


def main():
    # 1. Config
    config = ConfigLoader()

    # 2. Data Loading
    loader = DataLoader(config)
    print("Loading data...")
    movies_df = loader.load_movies()
    ratings_df = loader.load_ratings()

    tmdb_attr = loader.load_tmdb_attributes()
    tmdb_assoc = loader.load_tmdb_imdb_association()

    # Check if credits exist, if not download
    credits_path = loader._get_path("movie_credits_file")
    if not os.path.exists(credits_path):
        print("Credits file not found. Downloading...")
        # Need TMDBClient
        from src.tmdb import TMDBClient

        tmdb = TMDBClient(config)
        # We need tmdb_ids from association
        if not tmdb_assoc.empty and "tmdb_id" in tmdb_assoc.columns:
            # unique IDs
            ids = tmdb_assoc["tmdb_id"].unique().tolist()
            # Reduce wait time for faster download if possible or rely on config
            tmdb.download_credits(credits_path, ids)
        else:
            print(
                "Error: Cannot download credits, TMDB association missing or invalid."
            )
            return

    credits = loader.load_credits()

    # 3. Data Cleaning & Preparation
    cleaner = DataCleaner(config)
    movies_df = cleaner.preprocess_movies(movies_df)
    ratings_df = cleaner.preprocess_ratings(ratings_df, movies_df)
    movies_df = cleaner.filter_movies_by_ratings(movies_df, ratings_df)

    # Merge for Recommenders
    feature_df, flattened_URM = cleaner.prepare_recommender_data(
        movies_df, ratings_df, tmdb_attr, tmdb_assoc, credits
    )

    if flattened_URM.empty:
        print("Error: User Rating Matrix is empty after processing.")
        return

    # 4. Split Data
    test_size = config.get("recommender.test_size", 0.2)
    random_seed = config.get("recommender.random_seed", 42)

    print(f"Splitting data (test_size={test_size})...")
    train_set, validation_set = train_test_split(
        flattened_URM,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=flattened_URM["UserID"],
    )

    # Helper to get titles
    def get_movie_titles(movie_ids, feature_df):
        titles = []
        for mid in movie_ids:
            # feature_df has 'id' and 'title'
            match = feature_df[feature_df["id"] == mid]
            if not match.empty:
                titles.append(match.iloc[0]["title"])
            else:
                titles.append(f"Unknown ID ({mid})")
        return titles

    def print_user_history(user_id, ratings_df, feature_df, limit=10):
        print(f"\nUser {user_id} History (Top {limit} Rated):")
        user_hist = ratings_df[ratings_df["UserID"] == user_id].sort_values(
            by="Ratings", ascending=False
        )

        for _, row in user_hist.head(limit).iterrows():
            mid = row["id"]
            rating = row["Ratings"]
            match = feature_df[feature_df["id"] == mid]
            title = match.iloc[0]["title"] if not match.empty else f"ID {mid}"
            print(f" - {title} (Rating: {rating})")
        print("")

    # 5. Baseline Recommender
    print("\n--- Baseline Recommender (TopPop) ---")
    top_pop = BaselineRecommender(train_set, correcting_factor_metric="median")
    p5 = Evaluator.precision_at_k(validation_set, top_pop, k=5)
    print(f"Precision @ 5: {p5:.4f}")

    # Example recommendation
    user_id = validation_set["UserID"].iloc[0]

    # Print user history to verify relevance
    print_user_history(user_id, flattened_URM, feature_df, limit=10)

    recs = top_pop.recommend(user_id, k=5)
    print(f"Recommendations for User {user_id}:")
    rec_titles = get_movie_titles(recs, feature_df)
    for t in rec_titles:
        print(f" - {t}")

    # 6. Collaborative Filtering
    print("\n--- Collaborative Recommender (SVD) ---")
    collab = CollaborativeRecommender(train_set)
    p5_cf = Evaluator.precision_at_k(validation_set, collab, k=5)
    print(f"Precision @ 5: {p5_cf:.4f}")

    # Example
    recs_cf = collab.recommend(user_id, k=5)
    print(f"Recommendations (CF) for User {user_id}:")
    for t in get_movie_titles(recs_cf, feature_df):
        print(f" - {t}")

    # 7. Content-Based
    print("\n--- Content-Based Recommender ---")
    # Create Similarity Matrix
    print("Creating Feature Matrix...")
    sim_matrix = cleaner.create_feature_matrix(feature_df, include_overview=False)

    cb = ContentBasedRecommender(train_set, sim_matrix, feature_df)
    p5_cb = Evaluator.precision_at_k(validation_set, cb, k=5)
    print(f"Precision @ 5: {p5_cb:.4f}")

    # Example
    recs_cb = cb.recommend(user_id, k=5)
    print(f"Recommendations (CB) for User {user_id}:")
    for t in get_movie_titles(recs_cb, feature_df):
        print(f" - {t}")

    # 8. Hybrid Ensemble
    print("\n--- Hybrid Ensemble Recommender ---")
    ensemble = EnsembleRecommender(collab, cb)
    p5_ens = Evaluator.precision_at_k(validation_set, ensemble, k=5)
    print(f"Precision @ 5: {p5_ens:.4f}")

    # Example
    recs_ens = ensemble.recommend(user_id, k=5)
    print(f"Recommendations (Hybrid) for User {user_id}:")
    for t in get_movie_titles(recs_ens, feature_df):
        print(f" - {t}")

    # 9. Content-Based with Overview (Optional/Bonus)
    print("\n--- Content-Based (with Overview) ---")
    sim_matrix_ovw = cleaner.create_feature_matrix(feature_df, include_overview=True)
    cb_ovw = ContentBasedRecommender(train_set, sim_matrix_ovw, feature_df)
    p5_ovw = Evaluator.precision_at_k(validation_set, cb_ovw, k=5)
    print(f"Precision @ 5: {p5_ovw:.4f}")

    # Example
    recs_ovw = cb_ovw.recommend(user_id, k=5)
    print(f"Recommendations (CB with Overview) for User {user_id}:")
    for t in get_movie_titles(recs_ovw, feature_df):
        print(f" - {t}")

    # 10. Meta-Level Ensemble (using Overview CB)
    print("\n--- Meta-Level Ensemble ---")
    meta_ens = MetaEnsembleRecommender(collab, cb_ovw)
    p5_meta = Evaluator.precision_at_k(validation_set, meta_ens, k=5)
    print(f"Precision @ 5: {p5_meta:.4f}")

    # Example
    recs_meta = meta_ens.recommend(user_id, k=5)
    print(f"Recommendations (Meta) for User {user_id}:")
    for t in get_movie_titles(recs_meta, feature_df):
        print(f" - {t}")


if __name__ == "__main__":
    main()
