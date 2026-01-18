import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analyzer import DataAnalyzer
from src.cleaner import DataCleaner
from src.config import ConfigLoader
from src.loader import DataLoader
from src.tmdb import TMDBClient
from src.visualization import Visualizer


def main():
    # 1. Load Configuration
    config = ConfigLoader()

    # 2. Check Authentication
    print("Checking TMDB Authentication...")
    tmdb = TMDBClient(config)
    auth_status = tmdb.check_authentication()
    print(f"Auth Status: {auth_status}")

    # 3. Load Data
    loader = DataLoader(config)
    movies_df = loader.load_movies()
    ratings_df = loader.load_ratings()

    if movies_df.empty or ratings_df.empty:
        print("Error: One or more datasets could not be loaded. Exiting.")
        sys.exit(1)

    # 4. Clean Data
    cleaner = DataCleaner(config)

    # Clean Movies
    movies_df = cleaner.preprocess_movies(movies_df)

    # Clean Ratings (filter users and keep only relevant movies)
    ratings_df = cleaner.preprocess_ratings(ratings_df, movies_df)

    # Filter Movies again (keep only those with ratings)
    initial_movies_count = len(movies_df)
    movies_df = cleaner.filter_movies_by_ratings(movies_df, ratings_df)
    print(f"Filtered movies by ratings. Kept {len(movies_df)}/{initial_movies_count}.")

    # 5. Analysis & Visualization
    analyzer = DataAnalyzer(config)
    visualizer = Visualizer(config)

    # Genre Analysis
    print("\n--- Genre Analysis ---")
    genre_dist = analyzer.get_genre_distribution(movies_df)
    print(f"Number of genres: {len(genre_dist)}")
    print(genre_dist.head())
    visualizer.plot_genre_distribution(genre_dist)

    # Correlation
    print("\n--- Genre Correlation ---")
    corr_matrix = analyzer.calculate_genre_correlation(movies_df)
    visualizer.plot_correlation_matrix(corr_matrix)

    # Release Year Distribution
    print("\n--- Release Year ---")
    visualizer.plot_release_year_distribution(movies_df)

    # Ratings Analysis
    print("\n--- Ratings Analysis ---")
    merged_df, rating_counts, top_10_rated, ratings_per_user, user_stats = (
        analyzer.analyze_ratings(movies_df, ratings_df)
    )

    print("Top 10 rated movies (count of 10s):")
    print(top_10_rated)

    print("User Ratings Stats:")
    for k, v in user_stats.items():
        print(f"{k}: {v:.2f}")

    visualizer.plot_ratings_pie(rating_counts)
    visualizer.plot_ratings_per_user(ratings_per_user)

    # Temporal Analysis
    print("\n--- Temporal Analysis ---")
    temporal_df = analyzer.prepare_temporal_data(merged_df)
    visualizer.plot_ratings_over_time(temporal_df)

    # 6. Optional: TMDB Data Download (using TMDBClient)
    # This part can be triggered if data is missing or explicitly requested.
    # tmdb.download_data_by_movie_title(
    #     config.get("paths.tmdb_by_title_file"), movies_df["Title"]
    # )


if __name__ == "__main__":
    print("Starting Data Analysis Refactored Pipeline...")
    main()
