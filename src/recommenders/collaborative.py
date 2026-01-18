from typing import Any, List, Tuple, Union

import pandas as pd
import surprise
from surprise import Dataset, Reader

from .base import BaseRecommender


class CollaborativeRecommender(BaseRecommender):
    """Collaborative Filtering Recommender using the Surprise library (SVD).

    Attributes:
        train_set_df (pd.DataFrame): Training dataset.
        algo (surprise.AlgoBase): Trained Surprise algorithm.
    """

    def __init__(
        self,
        train_set_df: pd.DataFrame,
        rating_scale: Tuple[int, int] = (1, 10),
        algo: Any = None,
    ) -> None:
        """Initializes and trains the CF model.

        Args:
            train_set_df (pd.DataFrame): DataFrame with columns 'UserID', 'id', 'Ratings'.
            rating_scale (Tuple[int, int], optional): Min and Max ratings. Defaults to (1, 10).
            algo (Any, optional): Surprise algorithm instance. Defaults to SVD().
        """
        self.train_set_df = train_set_df
        # Ensure ID columns are consistent for lookup
        self.movie_catalog = train_set_df["id"].unique()
        self.user_profile = train_set_df.groupby("UserID")["id"].apply(list)

        # Prepare data for Surprise
        reader = Reader(rating_scale=rating_scale)
        # Surprise requires: user, item, rating
        data = Dataset.load_from_df(train_set_df[["UserID", "id", "Ratings"]], reader)
        self.surprise_train_set = data.build_full_trainset()

        # Train model
        self.algo = algo if algo else surprise.SVD()
        self.algo.fit(self.surprise_train_set)

    def recommend(self, user_id: int, k: int = 10):
        """Returns personalized recommendations using SVD.

        Args:
            user_id (int): User ID.
            k (int, optional): Number of recommendations. Defaults to 10.

        Returns:
            List[Union[int, str]]: List of movie IDs.
        """
        # Get movies user hasn't rated
        rated_movies = set(self.user_profile.get(user_id, []))

        # Predict ratings for all unrated movies
        # Note: For very large catalogs, this "predict all" approach is slow.
        # Optimized approach would use a candidate set or annoying indexing.
        # Given current scale, full scan is acceptable but slow.
        predictions = []
        for movie_id in self.movie_catalog:
            if movie_id not in rated_movies:
                pred = self.algo.predict(user_id, movie_id)
                predictions.append(pred)

        # Sort by estimated rating
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Filter movies with predicted rating > 5 (heuristics from original code)
        # and limit to k
        top_recommendations = [p.iid for p in predictions if p.est > 5][:k]

        # If filtering removed too many, fill with top remaining (optional improvement over original)
        if len(top_recommendations) < k:
            remaining = [p.iid for p in predictions if p.est <= 5][
                : k - len(top_recommendations)
            ]
            top_recommendations.extend(remaining)

        return top_recommendations
