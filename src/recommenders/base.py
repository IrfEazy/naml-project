from abc import ABC, abstractmethod
from typing import List, Union


class BaseRecommender(ABC):
    """Abstract base class for all recommender systems."""

    @abstractmethod
    def recommend(self, user_id: int, k: int = 10) -> List[Union[int, str]]:
        """Returns a list of top k recommended movie IDs for the given user.

        Args:
            user_id (int): The ID of the user to recommend for.
            k (int, optional): The number of recommendations to return. Defaults to 10.

        Returns:
            List[Union[int, str]]: List of recommended movie IDs.
        """
        pass
