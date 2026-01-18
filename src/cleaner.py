from collections import Counter
from typing import Any, List, Set, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import ConfigLoader


class DataCleaner:
    """Handles data preprocessing and cleaning for the movie recommender system.

    Attributes:
        config (ConfigLoader): Configuration loader instance.
    """

    def __init__(self, config: ConfigLoader) -> None:
        """Initializes the DataCleaner.

        Args:
            config (ConfigLoader): ConfigLoader instance.
        """
        self.config = config

    def preprocess_movies(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and preprocesses the movies dataframe.

        Steps:
        1. Drop duplicates.
        2. Split genres and filter invalid ones.
        3. Filter out rare genres.
        4. Extract Title and Year.
        5. Filter by release year range.

        Args:
            movies_df (pd.DataFrame): Raw movies dataframe.

        Returns:
            pd.DataFrame: Cleaned movies dataframe.
        """
        initial_count = len(movies_df)
        print("Preprocessing movies...")

        # Drop duplicates
        movies_df.drop_duplicates(inplace=True, ignore_index=True)

        # Split Genre: Handle strings, ensure lists, remove NA
        movies_df["Genre"] = movies_df["Genre"].astype(str).str.split("|")
        movies_df.dropna(subset=["Genre"], inplace=True)
        movies_df = movies_df[movies_df["Genre"].apply(lambda x: isinstance(x, list))]

        # Flatten genres to count frequencies
        all_genres = [g for genres in movies_df["Genre"].tolist() for g in genres]
        genre_counts = Counter(all_genres)
        threshold = self.config.get("analysis.popular_genre_threshold", 20)
        popular_genres: Set[str] = {
            genre for genre, count in genre_counts.items() if count >= threshold
        }

        # Filter movies keeping only popular genres
        movies_df = movies_df[
            movies_df["Genre"].apply(
                lambda genres: all(g in popular_genres for g in genres)
            )
        ].reset_index(drop=True)

        # Extract Title and Year from "MovieTitle(Year)" e.g., "Toy Story (1995)"
        if "MovieTitle(Year)" in movies_df.columns:
            extracted = movies_df["MovieTitle(Year)"].str.extract(r"(.*)\s+\((\d+)\)")
            movies_df["Title"] = extracted[0]
            movies_df["Release_year"] = pd.to_numeric(extracted[1], errors="coerce")
            movies_df.drop(columns=["MovieTitle(Year)"], inplace=True)

        # Filter by Year Range
        start_year = self.config.get("analysis.year_start", 2014)
        end_year = self.config.get("analysis.year_end", 2017)
        movies_df = movies_df[
            (movies_df["Release_year"] >= start_year)
            & (movies_df["Release_year"] <= end_year)
        ].reset_index(drop=True)

        print(
            f"Movies preprocessing done. Kept {len(movies_df)}/{initial_count} movies."
        )
        return movies_df

    def preprocess_ratings(
        self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filters ratings to include only valid movies and active users.

        Args:
            ratings_df (pd.DataFrame): Raw ratings dataframe.
            movies_df (pd.DataFrame): Cleaned movies dataframe.

        Returns:
            pd.DataFrame: Cleaned ratings dataframe.
        """
        initial_count = len(ratings_df)
        print("Preprocessing ratings...")

        # 1. Filter ratings for movies that exist in our cleaned movies_df
        valid_movie_ids = set(movies_df["MovieID"].unique())
        ratings_df = ratings_df[ratings_df["MovieID"].isin(valid_movie_ids)]

        # 2. Filter users with insufficient activity (Min Ratings)
        min_ratings = self.config.get("analysis.min_ratings_per_user", 20)
        # Calculate counts per user
        user_counts = ratings_df["UserID"].value_counts()
        valid_users = user_counts[user_counts >= min_ratings].index
        ratings_df = ratings_df[ratings_df["UserID"].isin(valid_users)].reset_index(
            drop=True
        )

        print(
            f"Ratings preprocessing done. Kept {len(ratings_df)}/{initial_count} ratings."
        )
        return ratings_df

    def filter_movies_by_ratings(
        self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filters movies to keep only those that have at least one rating.

        Args:
            movies_df (pd.DataFrame): Movies dataframe.
            ratings_df (pd.DataFrame): Ratings dataframe.

        Returns:
            pd.DataFrame: Filtered movies dataframe.
        """
        valid_movie_ids = set(ratings_df["MovieID"].unique())
        movies_df = movies_df[movies_df["MovieID"].isin(valid_movie_ids)].reset_index(
            drop=True
        )
        return movies_df

    def prepare_recommender_data(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        tmdb_attributes: pd.DataFrame,
        tmdb_association: pd.DataFrame,
        movie_credits: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Merges disparate datasets to create a unified feature set for recommenders.

        Args:
            movies_df (pd.DataFrame): Processed movies.
            ratings_df (pd.DataFrame): Processed ratings.
            tmdb_attributes (pd.DataFrame): Raw TMDB attributes.
            tmdb_association (pd.DataFrame): ID mapping (TMDB <-> IMDB).
            movie_credits (pd.DataFrame): Raw credits data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - full_movie_features: DataFrame with all metadata merged.
                - flattened_URM: DataFrame representing User-Item interactions with TMDB IDs.
        """
        print("Preparing data for recommenders...")

        # Filter attributes by association existence
        valid_tmdb_ids = set(tmdb_association["tmdb_id"])
        tmdb_attributes = tmdb_attributes[
            tmdb_attributes["id"].isin(valid_tmdb_ids)
        ].reset_index(drop=True)

        # Select and Helper to extract names from list of dicts
        def extract_names(x: Any) -> List[str]:
            if isinstance(x, list):
                return [d["name"] for d in x if isinstance(d, dict) and "name" in d]
            return []

        # Process metadata columns
        attributes_selected = tmdb_attributes[
            [
                "id",
                "title",
                "runtime",
                "original_language",
                "popularity",
                "budget",
                "production_companies",
                "production_countries",
                "overview",
            ]
        ].copy()

        attributes_selected["production_companies"] = attributes_selected[
            "production_companies"
        ].apply(extract_names)
        attributes_selected["production_countries"] = attributes_selected[
            "production_countries"
        ].apply(extract_names)
        attributes_selected.drop_duplicates(subset=["id"], inplace=True)

        # Process Credits
        movie_credits = (
            movie_credits[movie_credits["id"].isin(valid_tmdb_ids)]
            .drop_duplicates(subset=["id"])
            .reset_index(drop=True)
        )

        def extract_cast(x: Any) -> List[str]:
            if isinstance(x, list):
                return [d["name"] for d in x if isinstance(d, dict) and "name" in d]
            return []

        def extract_crew(x: Any, job: str) -> List[str]:
            if isinstance(x, list):
                return [
                    d["name"]
                    for d in x
                    if isinstance(d, dict) and d.get("known_for_department") == job
                ]
            return []

        movie_contribution = pd.DataFrame(
            {
                "id": movie_credits["id"],
                "actors": movie_credits["cast"].apply(extract_cast),
                "directors": (
                    movie_credits["crew"].apply(lambda x: extract_crew(x, "Directing"))
                ),
                "writers": (
                    movie_credits["crew"].apply(lambda x: extract_crew(x, "Writing"))
                ),
            }
        )

        # Merge Attributes + Credits
        tmdb_movies = pd.merge(
            attributes_selected, movie_contribution, on="id", how="inner"
        )

        # Prepare User-Rating Matrix mapped to TMDB IDs
        if "imdb_id" not in tmdb_association.columns:
            print("Warning: imdb_id not in association file")
            return tmdb_movies, pd.DataFrame()

        # Merge Ratings with Association to get TMDB IDs
        # Ratings has 'MovieID' which corresponds to 'imdb_id' in association
        user_rating_tmdb = pd.merge(
            ratings_df, tmdb_association, left_on="MovieID", right_on="imdb_id"
        ).drop(columns=["MovieID", "imdb_id"])

        flattened_URM = user_rating_tmdb[["UserID", "tmdb_id", "Ratings"]].rename(
            columns={"tmdb_id": "id"}
        )

        # Merge Movies.dat content (Genre, Year) into the Feature Matrix
        movies_with_tmdb = pd.merge(
            movies_df, tmdb_association, left_on="MovieID", right_on="imdb_id"
        )
        movies_with_tmdb.rename(columns={"tmdb_id": "id"}, inplace=True)
        movies_with_tmdb.drop(columns=["MovieID", "imdb_id"], inplace=True)

        # Final merge
        full_movie_features = pd.merge(movies_with_tmdb, tmdb_movies, on="id")

        if "Title" in full_movie_features.columns:
            full_movie_features.drop(columns=["Title"], inplace=True)

        return full_movie_features, flattened_URM

    def create_feature_matrix(
        self, movies_df: pd.DataFrame, include_overview: bool = False
    ) -> Any:
        """Creates a TF-IDF Similarity Matrix from movie metadata.

        Args:
            movies_df (pd.DataFrame): DataFrame containing movie features (Genre, actors, etc.).
            include_overview (bool): Whether to include the 'overview' text in the Features.

        Returns:
            np.ndarray: Cosine similarity matrix.
        """
        selected_features = [
            "Genre",
            "production_companies",
            "actors",
            "directors",
            "writers",
            "production_countries",
            "original_language",
        ]

        df = movies_df.copy()

        # Stringify list features for TF-IDF
        for feature in selected_features:
            if feature in df.columns:
                df[feature] = (
                    df[feature]
                    .fillna("")
                    .apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
                )

        # Combine into single string per movie
        combined_features = df[selected_features].agg(" ".join, axis=1).astype(str)

        if include_overview and "overview" in df.columns:
            overview = (
                df["overview"]
                .fillna("")
                .apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
            )
            combined_features = (
                combined_features.astype(str) + " " + overview.astype(str)
            )

        # Create TF-IDF Matrix
        vectorizer = TfidfVectorizer(stop_words="english")
        feature_matrix = vectorizer.fit_transform(combined_features.str.lower())

        # Compute Cosine Similarity
        similarity_matrix = cosine_similarity(feature_matrix)

        return similarity_matrix
