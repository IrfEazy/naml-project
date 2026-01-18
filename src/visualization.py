import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from src.config import ConfigLoader


class Visualizer:
    """Handles data visualization logic.

    Attributes:
        config (ConfigLoader): Configuration loader.
    """

    def __init__(self, config: ConfigLoader) -> None:
        """Initializes the Visualizer.

        Args:
            config (ConfigLoader): Configuration loader instance.
        """
        self.config = config

    def plot_genre_distribution(self, genre_df: pd.DataFrame) -> None:
        """Plots genre distribution (Pie chart).

        Args:
            genre_df (pd.DataFrame): Dataframe with 'Genre' and 'Count'.
        """
        fig = px.pie(
            genre_df, values="Count", names="Genre", title="Distribution of the Genres"
        )
        fig.show()

    def plot_correlation_matrix(self, correlation_matrix: pd.DataFrame) -> None:
        """Plots genre correlation matrix (Cluster Map).

        Args:
            correlation_matrix (pd.DataFrame): Genre correlation matrix.
        """
        plt.figure(figsize=(12, 10))
        g = sns.clustermap(correlation_matrix, center=0, cmap="vlag", figsize=(12, 12))
        if g.ax_row_dendrogram:
            g.ax_row_dendrogram.remove()
        plt.show()

    def plot_release_year_distribution(self, movies_df: pd.DataFrame) -> None:
        """Plots distribution of movies by release year.

        Args:
            movies_df (pd.DataFrame): Movies dataframe with 'Release_year'.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=movies_df, x="Release_year", discrete=True, element="step")
        plt.title("Number of Movies Released per Year")
        plt.show()

    def plot_ratings_pie(self, rating_counts: pd.Series) -> None:
        """Plots ratings distribution (Pie chart).

        Args:
            rating_counts (pd.Series): Counts of each rating value.
        """
        fig = px.pie(
            values=rating_counts.values,
            names=rating_counts.index,
            title="Distribution of the ratings' values",
        )
        fig.show()

    def plot_ratings_per_user(self, ratings_per_user: pd.Series) -> None:
        """Plots histogram of ratings per user.

        Args:
            ratings_per_user (pd.Series): Series of rating counts per user.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(x=ratings_per_user, binwidth=20, edgecolor="black")
        plt.title("Ratings per user")
        plt.show()

        # Zoomed in version
        plt.figure(figsize=(10, 6))
        sns.histplot(
            ratings_per_user,
            binwidth=20,
            binrange=(ratings_per_user.min(), 200),
            edgecolor="black",
        )
        plt.title("Ratings per user (Zoomed)")
        plt.show()

    def plot_ratings_over_time(self, temporal_df: pd.DataFrame) -> None:
        """Plots number of reviews per day for different release years.

        Args:
            temporal_df (pd.DataFrame): Dataframe merged from year-specific counts.
        """
        if temporal_df.empty:
            print("No temporal data to plot.")
            return

        fig = px.line(
            temporal_df,
            x="Date",
            y="Count",
            color="Release_Year",
            title="Number of reviews per day for movies between 2014 and 2017",
        )
        fig.show()
