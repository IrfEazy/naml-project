from typing import List, Optional, Union

import pandas as pd

from .base import BaseRecommender


class BaselineRecommender(BaseRecommender):
    """Most Popular Recommender System.

    Suggests the most highly rated movies in the dataset using a Bayesian average.
    """

    def __init__(
        self,
        flattened_urm: pd.DataFrame,
        correcting_factor: float = 20.0,
        correcting_factor_metric: Optional[str] = None,
    ) -> None:
        """Initializes the class with a flattened User-Rating Matrix (URM).

        Args:
            flattened_urm (pd.DataFrame): DataFrame with columns 'id' and 'Ratings'.
            correcting_factor (float, optional): Factor to adjust importance of items with few ratings. Defaults to 20.0.
            correcting_factor_metric (Optional[str], optional): 'avg' or 'median' to dynamically set correcting factor. Defaults to None.
        """
        self.flattened_urm = flattened_urm

        ratings_count = flattened_urm.groupby("id").size()

        # Dynamic correcting factor logic
        if correcting_factor_metric == "avg":
            correcting_factor = float(ratings_count.mean())
        elif correcting_factor_metric == "median":
            correcting_factor = float(ratings_count.median())

        # Calculate mean rating with Bayesian average smoothing
        # Sum of ratings / (count + factor)
        sum_ratings = flattened_urm.groupby("id")["Ratings"].sum()
        count_ratings = flattened_urm.groupby("id")["Ratings"].count()

        mean_rating_per_movie = sum_ratings / (count_ratings + correcting_factor)

        self.mean_rating_per_movie = mean_rating_per_movie.sort_values(ascending=False)

    def recommend(self, user_id: int, k: int = 10) -> List[Union[int, str]]:
        """Returns top k popular movies (non-personalized).

        Args:
            user_id (int): User ID (ignored for baseline).
            k (int, optional): Number of recommendations. Defaults to 10.

        Returns:
            List[Union[int, str]]: List of movie IDs.
        """
        return self.mean_rating_per_movie.head(k).index.tolist()
