from typing import List, Tuple, Union

from .base import BaseRecommender
from .collaborative import CollaborativeRecommender
from .content_based import ContentBasedRecommender


class EnsembleRecommender(BaseRecommender):
    """Hybrid recommender combining Collaborative Filtering and Content-Based.

    Uses an intersection strategy with a union fallback.

    Attributes:
        cf_model (CollaborativeRecommender): Collaborative Filtering model.
        cb_model (ContentBasedRecommender): Content-Based model.
    """

    def __init__(
        self,
        collaborative_model: CollaborativeRecommender,
        content_based_recommender: ContentBasedRecommender,
    ) -> None:
        """Initializes the Ensemble Recommender.

        Args:
            collaborative_model (CollaborativeRecommender): Instance of CollaborativeRecommender.
            content_based_recommender (ContentBasedRecommender): Instance of ContentBasedRecommender.
        """
        self.cf_model = collaborative_model
        self.cb_model = content_based_recommender

    def recommend(self, user_id: int, k: int = 10):
        """Returns recommendations based on intersection of CF and CB approaches.

        Args:
            user_id (int): User ID.
            k (int, optional): Number of recommendations. Defaults to 10.

        Returns:
            List[Union[int, str]]: List of movie IDs.
        """
        cf_ids = self.cf_model.recommend(user_id, k)
        cb_ids = self.cb_model.recommend(user_id, k)

        cf_set = set(cf_ids)
        cb_set = set(cb_ids)

        common_movies = list(cf_set & cb_set)

        if not common_movies:
            # Union fallback
            union_movies = list(cf_set | cb_set)
            scored_union: List[Tuple[int, float]] = []

            for mid in union_movies:
                # Use CB model similarity to "best representative" as a heuristic score?
                # Or just use the fact they were recommended.
                # Here we reuse the logic from the original: find nearest neighbor in CB space to get a 'quality' score
                recs = self.cb_model._get_movie_recommendations_by_movie(
                    mid, suggest_n=1
                )
                best_score = float(recs[0][2]) if recs else 0.0
                scored_union.append((mid, best_score))

            scored_union.sort(key=lambda x: x[1], reverse=True)
            common_movies = [m for m, _ in scored_union[:k]]
        else:
            if len(common_movies) < k:
                # Fill remaining spots with union sorted by same metric
                remaining_needed = k - len(common_movies)
                union_movies = list(cf_set | cb_set)

                scored_union = []
                for mid in union_movies:
                    if mid not in common_movies:
                        recs = self.cb_model._get_movie_recommendations_by_movie(
                            mid, suggest_n=1
                        )
                        best_score = float(recs[0][2]) if recs else 0.0
                        scored_union.append((mid, best_score))

                scored_union.sort(key=lambda x: x[1], reverse=True)
                common_movies.extend([m for m, _ in scored_union[:remaining_needed]])

        return common_movies


class MetaEnsembleRecommender(BaseRecommender):
    """Meta-level Hybrid.

    Uses CF recommendations as input to CB content recommendations to find "similar to what the crowd says I like".
    """

    def __init__(
        self,
        collaborative_model: CollaborativeRecommender,
        content_based_recommender: ContentBasedRecommender,
    ) -> None:
        """Initializes the Meta Ensemble.

        Args:
            collaborative_model (CollaborativeRecommender): CF model.
            content_based_recommender (ContentBasedRecommender): CB model.
        """
        self.cf_model = collaborative_model
        self.cb_model = content_based_recommender

    def recommend(self, user_id: int, k: int = 10) -> List[Union[int, str]]:
        """Generates recommendations by feeding CF results into CB finder.

        Args:
            user_id (int): User ID.
            k (int, optional): Number of recommendations. Defaults to 10.

        Returns:
            List[Union[int, str]]: List of movie IDs.
        """
        # 1. Get CF recommendations
        cf_ids = self.cf_model.recommend(user_id, k)

        ids: List[int] = []
        scores: List[float] = []

        # 2. For each CF recommendation, find similar items using CB
        for mid in cf_ids:
            cb_recs = self.cb_model._get_movie_recommendations_by_movie(
                mid, suggest_n=k
            )

            for rec_id, _, score in cb_recs:
                if rec_id not in ids:
                    ids.append(rec_id)
                    scores.append(score)

        # 3. Sort by score
        recommendations = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)

        return [mid for mid, _ in recommendations[:k]]
