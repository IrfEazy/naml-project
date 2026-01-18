from collections import Counter
from typing import Dict, Tuple

import pandas as pd

from src.config import ConfigLoader


class DataAnalyzer:
    """Performs statistical analysis on movie and rating data.

    Attributes:
        config (ConfigLoader): Configuration loader.
    """

    def __init__(self, config: ConfigLoader) -> None:
        """Initializes the DataAnalyzer.

        Args:
            config (ConfigLoader): Configuration loader instance.
        """
        self.config = config

    def get_genre_distribution(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates genre counts.

        Args:
            movies_df (pd.DataFrame): Movies dataframe with 'Genre' column (list of strings).

        Returns:
            pd.DataFrame: Dataframe with 'Genre' and 'Count' columns.
        """
        all_genres = [g for genres in movies_df["Genre"].values for g in genres]
        genre_counts = Counter(all_genres).most_common()
        return pd.DataFrame(genre_counts, columns=["Genre", "Count"])

    def calculate_genre_correlation(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Computes correlation matrix between genres.

        Args:
            movies_df (pd.DataFrame): Movies dataframe.

        Returns:
            pd.DataFrame: Correlation matrix of genres.
        """
        # Collect all unique genres
        all_genres = sorted(list({g for genres in movies_df["Genre"] for g in genres}))

        # Create binary matrix
        # Initialize with zeros
        genre_matrix = pd.DataFrame(0, index=movies_df.index, columns=all_genres)

        # Populate matrix (One hot encoding manual approach for lists)
        for idx, genres in enumerate(movies_df["Genre"]):
            genre_matrix.loc[idx, genres] = 1

        correlation_matrix = genre_matrix.corr()
        return correlation_matrix

    def analyze_ratings(
        self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, Dict[str, float]]:
        """Merges data and performs various rating analyses.

        Args:
            movies_df (pd.DataFrame): Movies df.
            ratings_df (pd.DataFrame): Ratings df.

        Returns:
            Tuple containing:
            - merged_df (pd.DataFrame): Merged movie and rating data.
            - rating_counts (pd.Series): Counts of each rating value.
            - top_10_rated (pd.Series): Top 10 movies with perfect 10 rating.
            - ratings_per_user (pd.Series): Count of ratings per user.
            - stats (Dict[str, float]): Descriptive statistics of user activity.
        """
        merged_df = movies_df.merge(ratings_df, on="MovieID")

        # Distribution of rating values
        rating_counts = merged_df["Ratings"].value_counts().sort_index()

        # Top movies with rating 10
        top_10_rated = (
            merged_df[merged_df["Ratings"] == 10]["Title"].value_counts().head(10)
        )

        # Ratings per user stats
        ratings_per_user = merged_df.groupby("UserID")["Ratings"].count()
        stats = {
            "mean": float(ratings_per_user.mean()),
            "median": float(ratings_per_user.median()),
            "mode": (
                float(ratings_per_user.mode()[0])
                if not ratings_per_user.mode().empty
                else 0.0
            ),
            "max": float(ratings_per_user.max()),
            "min": float(ratings_per_user.min()),
        }

        return merged_df, rating_counts, top_10_rated, ratings_per_user, stats

    def prepare_temporal_data(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Extracts date components and prepares time series data for analysis.

        Args:
            merged_df (pd.DataFrame): Merged movies and ratings dataframe.

        Returns:
            pd.DataFrame: Temporal data suitable for plotting (Count of reviews per day per Release Year).
        """
        df = merged_df.copy()
        df["datetime"] = pd.to_datetime(df["RatingTimestamp"], unit="s")
        df["Date"] = df["datetime"].dt.date
        df["Year"] = df["datetime"].dt.year

        temporal_data = []
        # Hardcoded range based on original analysis requirements (2014-2017)
        # Could be parameterized via config if needed.
        start_year = self.config.get("analysis.year_start", 2014)
        end_year = self.config.get("analysis.year_end", 2017)

        for year in range(start_year, end_year + 1):
            year_data = (
                df[df["Release_year"] == year]
                .groupby("Date")["UserID"]
                .count()
                .reset_index()
            )
            year_data.rename(columns={"UserID": "Count"}, inplace=True)
            year_data["Release_Year"] = year
            temporal_data.append(year_data)

        if temporal_data:
            return pd.concat(temporal_data)
        return pd.DataFrame()
