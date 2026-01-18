from typing import Any, List, Set

import pandas as pd
from tqdm import tqdm

from src.recommenders.base import BaseRecommender


class Evaluator:
    """Evaluator for Recommender Systems."""

    @staticmethod
    def precision_at_k(
        testset_df: pd.DataFrame,
        recommender_object: BaseRecommender,
        k: int = 10,
        relevance_threshold: int = 5,
        verbose: int = 1,
    ) -> float:
        """Computes Mean Precision at K for a test set.

        Precision@K = (Number of recommended items that are relevant) / K

        Args:
            testset_df (pd.DataFrame): Test dataset with 'UserID' and 'Ratings'.
            recommender_object (BaseRecommender): The recommender system model.
            k (int, optional): The cutoff rank for recommendations. Defaults to 10.
            relevance_threshold (int, optional): Rating threshold to consider an item relevant. Defaults to 5.
            verbose (int, optional): Verbosity level (0=silent, 1=progress bar). Defaults to 1.

        Returns:
            float: The mean Precision@K across all users in the test set.
        """
        test_users = testset_df["UserID"].unique()

        # Pre-compute relevant items for each user (Ground Truth)
        # Using a dictionary for O(1) lookups
        relevant_movies = (
            testset_df[testset_df["Ratings"] > relevance_threshold]
            .groupby("UserID")["id"]
            .apply(set)
            .to_dict()
        )

        precision_values: List[float] = []

        iterator = tqdm(
            test_users, disable=(verbose == 0), leave=False, desc="Evaluating"
        )

        for user_id in iterator:
            user_relevant_movies: Set[Any] = relevant_movies.get(user_id, set())

            # Skip users with no ground truth relevant items?
            # Original logic skipped if not relevant items.
            if not user_relevant_movies:
                continue

            try:
                # Get K recommendations
                recommendations = recommender_object.recommend(user_id, k)
            except Exception as e:
                print(f"Error recommending for user {user_id}: {e}")
                continue

            # Calculate Hits (Intersection of Recommendations and Ground Truth)
            hits = len(set(recommendations) & user_relevant_movies)

            # Precision Formula: Hits / K
            # We handle the case where the recommender returns fewer than K items
            denom = max(1, min(k, len(recommendations)))

            # Note: Standard P@K divides by K strictly. Relaxed handles variable length.
            precision_values.append(hits / denom)

        return (
            float(sum(precision_values) / len(precision_values))
            if precision_values
            else 0.0
        )
