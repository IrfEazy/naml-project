import json
import os
from typing import Any, Dict, List, Union

import pandas as pd

from src.config import ConfigLoader


class DataLoader:
    """Handles loading of data from various file formats.

    Attributes:
        config (ConfigLoader): Configuration loader instance.
        data_dir (str): Base directory for data files.
    """

    def __init__(self, config: ConfigLoader) -> None:
        """Initializes the DataLoader.

        Args:
            config (ConfigLoader): ConfigLoader instance.
        """
        self.config = config
        self.data_dir = self.config.get("paths.data_dir", "Data")

    def _get_path(self, file_key: str) -> str:
        """Resolves the absolute path for a given file key from config."""
        filename = self.config.get(f"paths.{file_key}")
        return os.path.join(self.data_dir, filename)

    def load_movies(self) -> pd.DataFrame:
        """Loads the movies dataset (DAT format).

        Returns:
            pd.DataFrame: Dataframe containing MovieID, MovieTitle(Year), and Genre.
        """
        path = self._get_path("movies_file")
        columns = ["MovieID", "MovieTitle(Year)", "Genre"]
        if os.path.exists(path):
            try:
                msg = f"Loading movies from {path}..."
                print(msg)
                return pd.read_csv(
                    path,
                    delimiter="::",
                    names=columns,
                    engine="python",
                    encoding="latin-1",
                )
            except Exception as e:
                print(f"Error loading movies: {e}")
                return pd.DataFrame(columns=columns)
        else:
            print(f"Warning: {path} not found.")
            return pd.DataFrame(columns=columns)

    def load_ratings(self) -> pd.DataFrame:
        """Loads the ratings dataset (DAT format).

        Returns:
            pd.DataFrame: Dataframe containing UserID, MovieID, Ratings, and Timestamp.
        """
        path = self._get_path("ratings_file")
        columns = ["UserID", "MovieID", "Ratings", "RatingTimestamp"]
        if os.path.exists(path):
            try:
                print(f"Loading ratings from {path}...")
                return pd.read_csv(
                    path,
                    delimiter="::",
                    names=columns,
                    engine="python",
                    encoding="latin-1",
                )
            except Exception as e:
                print(f"Error loading ratings: {e}")
                return pd.DataFrame(columns=columns)
        else:
            print(f"Warning: {path} not found.")
            return pd.DataFrame(columns=columns)

    def _load_json_as_df(self, file_key: str) -> pd.DataFrame:
        """Generic helper to load a JSON file into a DataFrame.

        Args:
            file_key (str): Configuration key for the filename.

        Returns:
            pd.DataFrame: Loaded data or empty DataFrame if not found.
        """
        path = self._get_path(file_key)
        if os.path.exists(path):
            try:
                return pd.read_json(path)
            except ValueError:
                # Handle cases where JSON might be a list of primitives or mixed
                with open(path, "r") as f:
                    data = json.load(f)
                return pd.DataFrame(data)
        return pd.DataFrame()

    def load_tmdb_movies(self) -> pd.DataFrame:
        """Loads TMDB movies JSON data."""
        return self._load_json_as_df("tmdb_movies_file")

    def load_tmdb_attributes(self) -> pd.DataFrame:
        """Loads detailed TMDB attributes (budget, revenue, etc.)."""
        return self._load_json_as_df("movie_tmdb_attributes_file")

    def load_tmdb_imdb_association(self) -> pd.DataFrame:
        """Loads the mapping between TMDB IDs and IMDB IDs."""
        return self._load_json_as_df("tmdb_imdb_association_file")

    def load_credits(self) -> pd.DataFrame:
        """Loads movie credits (cast and crew)."""
        return self._load_json_as_df("movie_credits_file")

    def load_json(self, file_key: str) -> Union[List[Any], Dict[str, Any]]:
        """Generic loader for raw JSON data (returns list or dict).

        Args:
            file_key (str): Configuration key for the filename.

        Returns:
            Union[List[Any], Dict[str, Any]]: Parsed JSON data or empty list on failure.
        """
        path = self._get_path(file_key)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON {path}: {e}")
                return []
        else:
            print(f"Warning: {path} not found.")
            return []
