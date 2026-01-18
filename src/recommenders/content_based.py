from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from .base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """Content-Based Recommender using TF-IDF and Cosine Similarity.

    Attributes:
        flattened_urm (pd.DataFrame): User-Rating Matrix.
        similarity_matrix (np.ndarray): Pre-computed cosine similarity matrix.
        data_movies (pd.DataFrame): Movies metadata.
        correcting_factor (float): Factor to adjust weighted scores.
        movie_id_to_index (Dict[int, int]): Mapping from Movie ID to Matrix Index.
    """

    def __init__(
        self,
        flattened_urm: pd.DataFrame,
        similarity_matrix: Any,
        data_movies: pd.DataFrame,
        correcting_factor: float = 0.2,
    ) -> None:
        """Initializes the Content-Based Recommender.

        Args:
            flattened_urm (pd.DataFrame): DataFrame of user ratings ['UserID', 'id', 'Ratings'].
            similarity_matrix (Any): Pre-computed cosine similarity matrix (numpy array or sparse).
            data_movies (pd.DataFrame): DataFrame of movies ['id', 'title', ...].
            correcting_factor (float, optional): Factor to adjust weighted scores. Defaults to 0.2.
        """
        self.flattened_urm = flattened_urm
        self.similarity_matrix = similarity_matrix
        self.data_movies = data_movies
        self.correcting_factor = correcting_factor

        # Map Movie ID to matrix index
        self.movie_id_to_index: Dict[int, int] = {
            mid: idx for idx, mid in zip(data_movies.index, data_movies["id"])
        }

    def _get_movie_recommendations_by_movie(
        self, movie_id: int, suggest_n: int = 10
    ) -> List[Tuple[int, str, float]]:
        """Finds similar movies to a given movie ID.

        Args:
            movie_id (int): Target movie ID.
            suggest_n (int, optional): Number of similar movies to return. Defaults to 10.

        Returns:
            List[Tuple[int, str, float]]: List of (rec_id, title, score).
        """
        if movie_id not in self.movie_id_to_index:
            return []

        idx = self.movie_id_to_index[movie_id]
        similarity_scores = self.similarity_matrix[idx]

        # Sort indices by similarity descending
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # Top similar (skipping self at index 0 typically)
        top_indices = sorted_indices[1 : suggest_n + 1]

        recommendations = []
        for i in top_indices:
            rec_id = self.data_movies.iloc[i]["id"]
            title_col = "title" if "title" in self.data_movies.columns else "Title"
            rec_title = self.data_movies.iloc[i].get(title_col, f"ID {rec_id}")
            score = similarity_scores[i]
            recommendations.append((rec_id, str(rec_title), float(score)))

        return recommendations

    def recommend(self, user_id: int, k: int = 5) -> List[Union[int, str]]:
        """Generates recommendations based on the user's history and item similarity.

        Args:
            user_id (int): User ID.
            k (int, optional): Number of recommendations. Defaults to 5.

        Returns:
            List[Union[int, str]]: List of recommended movie IDs.
        """
        user_history = self.flattened_urm[self.flattened_urm["UserID"] == user_id]
        user_movie_ids = user_history["id"].tolist()
        user_ratings = user_history["Ratings"].tolist()

        if not user_movie_ids:
            return []

        # Aggregate recommendations
        scores: Dict[int, float] = defaultdict(float)
        weighted_scores: Dict[int, float] = defaultdict(float)

        for movie_id, rating in zip(user_movie_ids, user_ratings):
            recs = self._get_movie_recommendations_by_movie(movie_id, suggest_n=k)

            for rec_id, _, score in recs:
                if rec_id not in user_movie_ids:
                    scores[rec_id] += score
                    weighted_scores[rec_id] += score * rating

        final_recs = []
        for mid, score_sum in scores.items():
            # Weighted average rating prediction
            predicted_score = weighted_scores[mid] / (
                score_sum + self.correcting_factor
            )
            final_recs.append((mid, predicted_score))

        final_recs.sort(key=lambda x: x[1], reverse=True)

        return [mid for mid, _ in final_recs[:k]]


class ContentBasedWithFilteringRecommender(ContentBasedRecommender):
    """Content-Based Recommender that filters source movies by a rating threshold."""

    def __init__(
        self,
        flattened_urm: pd.DataFrame,
        similarity_matrix: Any,
        data_movies: pd.DataFrame,
        min_rating: int = 7,
    ) -> None:
        """Initializes the filtered recommender.

        Args:
            flattened_urm (pd.DataFrame): User-Rating Matrix.
            similarity_matrix (Any): Cosine similarity matrix.
            data_movies (pd.DataFrame): Movies metadata.
            min_rating (int, optional): Minimum rating to consider a movie for profile. Defaults to 7.
        """
        super().__init__(flattened_urm, similarity_matrix, data_movies)
        self.min_rating = min_rating

    def recommend(self, user_id: int, k: int = 5) -> List[Union[int, str]]:
        """Generates recommendations using only highly rated movies from user history.

        Args:
            user_id (int): User ID.
            k (int, optional): Number of recommendations. Defaults to 5.

        Returns:
            List[Union[int, str]]: List of recommended movie IDs.
        """
        user_history = self.flattened_urm[self.flattened_urm["UserID"] == user_id]

        # Filter by min_rating
        filtered_history = user_history[user_history["Ratings"] > self.min_rating]
        user_movie_ids = filtered_history["id"].tolist()

        if not user_movie_ids:
            return []

        final_scores: Dict[int, float] = defaultdict(float)
        repeated_counts: Dict[int, int] = defaultdict(int)

        # Doubling suggest_n as per original logic heuristics
        search_k = k * 2

        for movie_id in user_movie_ids:
            recs = self._get_movie_recommendations_by_movie(
                movie_id, suggest_n=search_k
            )

            for rec_id, _, score in recs:
                full_history_ids = user_history["id"].tolist()
                if rec_id not in full_history_ids:
                    final_scores[rec_id] += score
                    repeated_counts[rec_id] += 1

        final_recs = []
        for mid, total_score in final_scores.items():
            avg_score = total_score / repeated_counts[mid]
            final_recs.append((mid, avg_score))

        final_recs.sort(key=lambda x: x[1], reverse=True)

        return [mid for mid, _ in final_recs[:k]]
