# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [markdown] Cell 1
# # Data preprocessing


# %% [code] Cell 2
def is_colab():
    try:
        import google.colab

        return True
    except ImportError:
        return False


if is_colab():
    from google.colab import drive

    drive.mount("/content/drive")

# %% [code] Cell 3
import subprocess
import sys

try:
    __import__("surprise")
    print("surprise is already installed.")
except ImportError:
    print(f"surprise is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-surprise"])

# %% [code] Cell 4
import os
import random
import time

import numpy as np
import pandas as pd
import requests
import seaborn as sns
import surprise
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# fix randomness
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# %% [code] Cell 5
# Load and process movies data
# work_directory = './'
work_directory = "/content/drive/MyDrive/Colab Notebooks/NAML_Labs/NAML_Project/"
movies_path = work_directory + "Data/movies.dat"
ratings_path = work_directory + "Data/ratings.dat"
movies_columns = ["MovieID", "MovieTitle(Year)", "Genre"]
ratings_columns = ["UserID", "MovieID", "Ratings", "RatingTimestamp"]

# Read and process the movies data
data_movies = pd.read_csv(
    movies_path, delimiter="::", names=movies_columns, engine="python"
)
data_movies[["Title", "Release_year"]] = data_movies["MovieTitle(Year)"].str.extract(
    r"(.*)\s+\((\d+)\)"
)
data_movies.drop(columns=["MovieTitle(Year)"], inplace=True)
data_movies["Release_year"] = pd.to_numeric(data_movies["Release_year"])
data_movies = (
    data_movies.query("2014 <= Release_year <= 2017")
    .drop_duplicates()
    .reset_index(drop=True)
)
data_movies["Genre"] = data_movies["Genre"].str.split("|")
data_movies.dropna(subset=["Genre"], inplace=True)

# Read and filter the ratings data
data_ratings = pd.read_csv(
    ratings_path, delimiter="::", names=ratings_columns, engine="python"
)
data_ratings = (
    data_ratings[data_ratings["MovieID"].isin(data_movies["MovieID"])]
    .groupby("UserID")
    .filter(lambda x: len(x) >= 20)
    .reset_index(drop=True)
)

# Update data_movies to keep only movies present in filtered data_ratings
data_movies = data_movies[
    data_movies["MovieID"].isin(data_ratings["MovieID"])
].reset_index(drop=True)

# %% [code] Cell 6
# Load movie attributes and association data
movie_attributes = pd.read_json(
    os.path.join(work_directory, "Data/movie_tmdb_attributes.json")
)
tmdb_imdb_association = pd.read_json(
    os.path.join(work_directory, "Data/tmdb_imdb_association.json")
)

# Filter movies present in the association table
movie_attributes = movie_attributes[
    movie_attributes["id"].isin(tmdb_imdb_association["tmdb_id"])
].reset_index(drop=True)

# Select and transform the necessary columns
movie_attributes_selected = movie_attributes[
    ["id", "title", "runtime", "original_language", "popularity", "budget"]
].copy()
movie_attributes_selected["production_companies"] = movie_attributes[
    "production_companies"
].apply(lambda x: [d["name"] for d in x])
movie_attributes_selected["production_countries"] = movie_attributes[
    "production_countries"
].apply(lambda x: [d["name"] for d in x])
# drop duplicates by id
movie_attributes_selected.drop_duplicates(subset=["id"], inplace=True)

# Load movie credits data
movie_credits = pd.read_json(os.path.join(work_directory, "Data/movie_credits.json"))

# Filter and remove duplicates by 'id'
movie_credits = (
    movie_credits[movie_credits["id"].isin(tmdb_imdb_association["tmdb_id"])]
    .drop_duplicates(subset=["id"])
    .reset_index(drop=True)
)

# Extract relevant cast and crew information
movie_contribution = pd.DataFrame(
    {
        "id": movie_credits["id"],
        "actors": movie_credits["cast"].apply(lambda x: [d["name"] for d in x]),
        "actors_popularity": (
            movie_credits["cast"].apply(lambda x: [d["popularity"] for d in x])
        ),
        "directors": (
            movie_credits["crew"].apply(
                lambda x: [
                    d["name"] for d in x if d["known_for_department"] == "Directing"
                ]
            )
        ),
        "writers": (
            movie_credits["crew"].apply(
                lambda x: [
                    d["name"] for d in x if d["known_for_department"] == "Writing"
                ]
            )
        ),
    }
)

# Merge the movie attributes and contribution data on 'id'
tmdb_movies = pd.merge(
    movie_attributes_selected, movie_contribution, on="id", how="inner"
)
tmdb_movies.shape

# %% [code] Cell 7
ground_truth_path = work_directory + "Data/groundtruth.json"
api_key = "2d1b8795af328fd67395ac695c841792"

# %% [markdown] Cell 8
# # Evaluation Functions


# %% [code] Cell 9
def precision_at_k(
    testset_df,
    recommender_object,
    k=10,
    relevance_threshold=5,
    verbose=1,
    users_to_monitor=100,
):
    """
    Computes the precision at K for a recommender system over a test set.

    Precision at K measures how many of the top K recommended items are
    relevant, where relevance is determined by a threshold on the true ratings
    (i.e., items rated higher than the threshold are considered relevant).

    Parameters:
    -----------
    testset_df : pandas.DataFrame
        The test set containing actual ratings from users. The DataFrame should
        have the following columns:
        - 'UserID': User IDs
        - 'id': Item IDs (e.g., movie IDs)
        - 'Rarings': True ratings given by the user (ground truth).

    recommender_object : object
        The recommender system object. This object should have a
        `recommend(user_id, k)` method, which returns a list of the top K
        recommended items for the given user.

    k : int, optional (default=10)
        The number of top items to consider for calculating precision. For
        example, if K=10, precision will be calculated for the top 10
        recommendations.

    relevance_threshold : int or float, optional (default=5)
        The rating threshold above which items are considered relevant. For
        instance, if the true rating (r_ui) for an item is greater than or
        equal to `relevance_threshold`, the item is considered relevant.

    verbose : int, optional (default=1)
        Controls whether to print progress information during the computation.
        - If verbose=1, prints progress updates.
        - If verbose=0, no output will be printed.

    Returns:
    --------
    mean_precision : float
        The average precision at K across all users in the test set.

    Notes:
    ------
    - Precision at K is calculated for each user by dividing the number of
      relevant items in the top K recommendations by K (or fewer, if fewer than K
      items are recommended).
    - The precision scores for all users are averaged to compute the final
      result.
    """

    test_users = testset_df["UserID"].unique()
    n_users = test_users.shape[0]

    relevant_movies = (
        testset_df[testset_df["Ratings"] > relevance_threshold]
        .groupby("UserID")["id"]
        .apply(list)
        .to_dict()
    )
    precision_values = []

    # variables to monitor the time required for the processing
    counter = 0
    avg_processing_time = 0
    start_time = time.time()

    # Iterate over each user in the test set
    for user_id in test_users:

        counter += 1
        if verbose > 0 and counter % users_to_monitor == 0:
            elapsed_time = time.time() - start_time
            avg_processing_time = elapsed_time / (counter + 1)
            estimated_time = (n_users - counter) * avg_processing_time
            print(
                f"{counter:5d} users processed in {elapsed_time//60:2.0f}m {elapsed_time%60:2.0f}s. "
                f"Estimated time to completion: {estimated_time//60:2.0f}m {estimated_time%60:2.0f}s"
            )

        # Get a list  of all the movies the user has not rated
        # Consider only movies rated above a rating threshold
        user_relevant_movies = relevant_movies.get(user_id, [])

        if not user_relevant_movies:
            continue

        # Predict the ratings for the new movies
        try:
            top_k_recommendations = recommender_object.recommend(user_id, k)
        except Exception as e:
            print(f"Error recommending for user {user_id}: {e}")
            continue
        # Calculate precision for the user
        precision = len(set(top_k_recommendations) & set(user_relevant_movies)) / max(
            1, min(k, len(top_k_recommendations))
        )
        precision_values.append(precision)

    # Average precision over all users
    mean_precision = np.mean(precision_values)

    return mean_precision


# a simpler way to compute precision;
# just pass the list of recommendations and relevant
# movies for each user in the testset
def precision(recommendations, relevants):
    """
    recommendations: a list f recommended items for each user
    relevants: a list of relevant items for each user
    """
    precision = 0
    if len(recommendations) > 0 and len(relevants) > 0:
        precision = len(set(recommendations) & set(relevants)) / len(recommendations)

    return precision


# %% [markdown] Cell 10
# # CF Recommender System

# %% [markdown] Cell 11
# ## Data Preparation for CF

# %% [code] Cell 12
# Merge the user_rating with the tmdb_movie_id and keep only the TMDB_movie_id as unique identifier of the movie
user_rating_tmdb = pd.merge(
    data_ratings, tmdb_imdb_association, left_on="MovieID", right_on="imdb_id"
).drop(["MovieID", "imdb_id"], axis=1)

# Keep only the data needed for the CF recommender system
flattened_URM = user_rating_tmdb[["UserID", "tmdb_id", "Ratings"]]
flattened_URM = flattened_URM.rename(columns={"tmdb_id": "id"})

# Merge tmdb_movies and the user_ratings
movie_user_data = pd.merge(flattened_URM, tmdb_movies, on="id")

# %% [code] Cell 13
from sklearn.model_selection import train_test_split

train_set, validation_set = train_test_split(
    flattened_URM,
    test_size=0.2,
    random_state=RANDOM_SEED,
    shuffle=True,
    stratify=flattened_URM["UserID"],
)

# %% [markdown] Cell 14
# ## Most popular Recommender
#
# A recommender system that suggest the most highly rated movies in the dataset (movie with the highest average rating).
#

# %% [markdown] Cell 15
# We are testing the precision function on a recsys which always suggest the top rated movies in the DB.


# %% [code] Cell 16
class TopPopRecommender:

    def __init__(
        self, flattened_URM, correcting_factor=20, correcting_factor_metric=None
    ):
        """
        Initializes the class with a flattened User-Rating Matrix (URM) and a correcting factor.

        Parameters:
        -----------
        flattened_URM : pandas.DataFrame
            A flattened URM containing user ratings for various items (e.g., movies, products).
            The DataFrame should have at least two columns: 'id' (the item ID) and 'Ratings' (the user ratings).
        correcting_factor : int, optional (default=20)
            A factor used to adjust the importance of items with few ratings. Defaults to the 20.
        """

        ratings_count = flattened_URM.groupby("id").size()

        # a correcting factor to lower the importance of movies
        # with few ratings
        if correcting_factor_metric == "avg":
            correcting_factor = ratings_count.mean()
        elif correcting_factor_metric == "median":
            correcting_factor = ratings_count.median()

        mean_rating_per_movie = flattened_URM.groupby("id")["Ratings"].apply(
            lambda x: sum(x) / (len(x) + correcting_factor)
        )

        self.mean_rating_per_movie = mean_rating_per_movie.sort_values(ascending=False)

    def recommend(self, user_id, k):
        # the user_id is required as input to conform to the reccomender_object
        # used in the precision function, but is not functional to how the
        # suggestion are produced. The recommendations are the same for every user
        return self.mean_rating_per_movie.head(k).keys().tolist()


# %% [code] Cell 17
# check that top_pop_recsys works
user = 1
top_pop_recsys = TopPopRecommender(train_set)
top5 = top_pop_recsys.recommend(user, 5)

print("\nMost Popular Recommendations:")
print(
    "\n".join(
        f"{i + 1}.\t{tmdb_movies[tmdb_movies['id'] == id]['title'].values[0]}"
        for i, id in enumerate(top5)
    )
)
print("\n")

# %% [markdown] Cell 18
# The ouput seems quite reasonable, all quite popular movies.

# %% [markdown] Cell 19
# Next, we will adjust the correction factor to observe its impact on the recommendations.

# %% [code] Cell 20
# experiment with the correcting factor
top_pop_recsys = TopPopRecommender(train_set, correcting_factor=0)
top5 = top_pop_recsys.recommend(user, 5)

print("\nMost Popular Recommendations:")
print(
    "\n".join(
        f"{i + 1}.\t{tmdb_movies[tmdb_movies['id'] == id]['title'].values[0]}"
        for i, id in enumerate(top5)
    )
)
print("\n")

# %% [markdown] Cell 21
# If we do not use a correcting factor we get recommended unknown movies; The system is biased towards movies that have received the highest ratign by very few users.

# %% [markdown] Cell 22
# Now, let's test different correction factors to evaluate their impact.

# %% [code] Cell 23
# experiment with the correcting factor
# use the mean as the correcting factor
top_pop_recsys = TopPopRecommender(train_set, correcting_factor_metric="avg")
top5 = top_pop_recsys.recommend(1, 5)

print("\nMost Popular Recommendations:")
print(
    "\n".join(
        f"{i + 1}.\t{tmdb_movies[tmdb_movies['id'] == id]['title'].values[0]}"
        for i, id in enumerate(top5)
    )
)
print("\n")

# %% [code] Cell 24
# experiment with the correcting factor
# use the median as the correcting factor
top_pop_recsys = TopPopRecommender(train_set, correcting_factor_metric="median")
top5 = top_pop_recsys.recommend(1, 5)

print("\nMost Popular Recommendations:")
print(
    "\n".join(
        f"{i + 1}.\t{tmdb_movies[tmdb_movies['id'] == id]['title'].values[0]}"
        for i, id in enumerate(top5)
    )
)
print("\n")

# %% [markdown] Cell 25
# ## Test precision function

# %% [markdown] Cell 26
# First, we create a fictional test set where each user has at least one of the top 5 highest-rated movies as a favorite (rating above the threshold).

# %% [code] Cell 27
top5 = top_pop_recsys.recommend(1, 5)
threshold = 5

fict_test_set = validation_set.loc[
    validation_set["id"].isin(top5) & (validation_set["Ratings"] > threshold)
]

# %% [markdown] Cell 28
# The way fict_test_set is construted gurantees that the precision @ 5 is at least 0.2, as there is always one relevant movie (in the top5) for each user. Additionally, it's unlikely that all relevant movies for a user that are also in the top 5 would appear in the validation set and so we expect the precision to be close to 0.2.

# %% [code] Cell 29
precision = precision_at_k(fict_test_set, top_pop_recsys, 5)

print(f"\nPrecision @ 5: {precision:.4f}")

# %% [markdown] Cell 30
# Using the same reasoning as for precision@5, the precision@10 should be approximately 0.1, since there is still one relevant movie for each user among the top 10 recommendations

# %% [code] Cell 31
precision = precision_at_k(fict_test_set, top_pop_recsys, 10)

print(f"\nPrecision @ 5: {precision:.4f}")

# %% [markdown] Cell 32
# ## TopPopRecommender baseline

# %% [markdown] Cell 33
# We can proceed to look at the precion @ 5 and @ 10 on the whole validation set.

# %% [code] Cell 34
print(
    f"Precision @ 5: {precision_at_k(validation_set, top_pop_recsys, k=5, relevance_threshold=5, verbose=0):.4f}"
)
print(
    f"Precision @ 10: {precision_at_k(validation_set, top_pop_recsys, k=10, relevance_threshold=5, verbose=0):.4f}"
)

# %% [markdown] Cell 35
# ## SVD model - with Surprise libray

# %% [code] Cell 36
# Load the data into a Surprise Dataset object
reader = Reader(rating_scale=(1, 10))
surprise_train_set = Dataset.load_from_df(
    train_set[["UserID", "id", "Ratings"]], reader
).build_full_trainset()

surprise_validation_set = (
    Dataset.load_from_df(validation_set[["UserID", "id", "Ratings"]], reader)
    .build_full_trainset()
    .build_testset()
)

# Use SVD algorithm to train the model
# use surprise random predictor
algo = surprise.SVD()

# Predict a movie with the trained model
algo.fit(surprise_train_set)
predictions = algo.test(surprise_validation_set)

accuracy.rmse(predictions)
accuracy.fcp(predictions)

# %% [markdown] Cell 37
# We create the recommender object to encapsulate the logic of the model trained with the surprise library.


# %% [code] Cell 38
class SurpriseModel:
    """
    A class to wrap a recommendation model and provide movie recommendations.

    This class uses a recommendation model (e.g., from the `surprise` library)
    to generate movie recommendations for a given user. It also requires a movie
    catalog, which is a list of all movies in the trainset.

    Attributes:
    -----------
    model : object
        The recommendation model used to predict ratings. It should have a
        `predict(user_id, item_id)` method.

    movie_catalog : list
        A list of all movie IDs available for recommendation. The model will
        generate predictions for these movies.

    Methods:
    --------
    recommend(user_id, k):
        Returns the top `k` movie recommendations for the specified user.
    """

    def __init__(self, model, train_set, relevance_threshold=5):
        self.model = model
        self.train_set = train_set
        self.movie_catalog = train_set["id"].unique()
        self.user_profile = train_set.groupby("UserID")["id"].apply(list)
        self.relevance_threshold = relevance_threshold

    def recommend(self, user_id, k):
        # Get a list of all the movies the user has not rated
        predictions = [
            self.model.predict(user_id, movie_id)
            for movie_id in self.movie_catalog
            if movie_id not in self.user_profile[user_id]
        ]

        # Sort the predictions by estimated rating
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Get the top
        top_recommendations = [
            prediction.iid for prediction in predictions[:k] if prediction.est > 5
        ]

        return top_recommendations


# %% [code] Cell 39
def user_rated_movies_list_print(user, users_watched_list):
    """
    Prints a list of movies rated by a specific user, sorted by rating in descending order.

    :param int user: The UserID of the user whose rated movies are to be printed.
    :param pandas.DataFrame users_watched_list: DataFrame containing user-movie interactions, including 'UserID', 'id' (movie ID), 'title', and 'Ratings'.

    :return: None
    """
    # Print the user rated movies
    movies = users_watched_list.query("UserID == @user").sort_values(
        by="Ratings", ascending=False
    )
    print(f"User {user} has rated {len(movies)} movies:")
    print(
        "\n".join(
            f"·\t{movie['title']}\t({movie['Ratings']})"
            for _, movie in movies.iterrows()
        )
    )


# %% [code] Cell 40
surprise_model = SurpriseModel(algo, train_set)

user_rated_movies_list_print(116, movie_user_data)
res = surprise_model.recommend(116, 10)

print("\nSurprise Recommendations:")
print(
    "\n".join(
        f"{i + 1}.\t{tmdb_movies[tmdb_movies['id'] == id]['title'].values[0]}"
        for i, id in enumerate(res)
    )
)
print("\n")

# %% [code] Cell 41
print(
    f"\nPrecision @ 5: {precision_at_k(validation_set, surprise_model, k=5, relevance_threshold=5, users_to_monitor=100):.4f}"
)

# %% [code] Cell 42
print(
    f"\nPrecision @ 10: {precision_at_k(validation_set, surprise_model, k=10, relevance_threshold=5, users_to_monitor=100):.4f}"
)

# %% [markdown] Cell 43
# # CB Recommender System

# %% [markdown] Cell 44
# ##  Prepare the data for a simple CB recommender system

# %% [code] Cell 45
data_movies.info()
data_movies.head()

# %% [code] Cell 46
tmdb_movies.info()
tmdb_movies.head()

# %% [code] Cell 47
tmdb_imdb_association.info()
tmdb_imdb_association.head()

# %% [code] Cell 48
# Merge data movies with tmdb_associations
data_movies = (
    pd.merge(data_movies, tmdb_imdb_association, left_on="MovieID", right_on="imdb_id")
    .rename(columns={"tmdb_id": "id"})
    .drop(columns=["MovieID", "imdb_id"])
)

data_movies.info()
data_movies.head()

# %% [code] Cell 49
# Merge tmdb_movies_MovieID and data_movies to get the genres
data_movies = pd.merge(data_movies, tmdb_movies, on="id").drop("Title", axis=1)

data_movies.info()
data_movies.head()

# %% [code] Cell 50
# We can create a single feature vector containing all features
selected_features = [
    "Genre",
    "production_companies",
    "actors",
    "directors",
    "writers",
    "production_countries",
    "original_language",
]

# Handling missing values and converting to strings
data_movies[selected_features] = (
    data_movies[selected_features]
    .fillna("")
    .applymap(lambda x: " ".join(x if isinstance(x, list) else [str(x)]))
)

# Combining selected features into a single feature vector
# combined_features keeps a comprehensive representation of each movie's characteristics
# expand the features that are lists in string
combined_features = data_movies[selected_features].agg(" ".join, axis=1)
combined_features

# %% [code] Cell 51
combined_features[0]

# %% [code] Cell 52
# Converting Text Data to Feature Vectors
# Creating an instance of TfidfVectorizer
vectorizer = TfidfVectorizer()

# Converting the combined features into feature vectors
feature_vectors = vectorizer.fit_transform(combined_features)

# Printing the feature vectors
print("Feature vectors:\n", feature_vectors)

# %% [code] Cell 53
feature_vectors.shape

# %% [markdown] Cell 54
# ## CB using the sklearn library

# %% [code] Cell 55
# Create the similarity matrix
# Getting the similarity scores using cosine similarity
cosine_sim = cosine_similarity(feature_vectors)
print(type(cosine_sim))

print(cosine_sim)

# Printing the shape of the similarity matrix
print("Shape of the similarity matrix:", cosine_sim.shape)


# %% [code] Cell 56
def most_similar_word(x, l):
    """
    Finds the word in the list `l` that is most similar to the string `x` using the LCS algorithm.

    :param x: The string to compare (user's input movie title)
    :param l: List of strings to compare against (list of movie titles)
    :return: The word from the list `l` that has the highest similarity score to `x`
    """

    # Function to calculate the LCS ratio similarity between two strings
    def lcs_similarity(w1, w2):
        """
        Calculates the length of the longest common subsequence (LCS) between two strings.

        :param w1: First string (e.g., user's movie title)
        :param w2: Second string (e.g., movie title from the list)
        :return: Length of the LCS between the two strings
        """
        m = len(w1)
        n = len(w2)
        L = [[0 for x in range(n + 1)] for x in range(m + 1)]

        # Building the matrix in bottom-up way
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif w1[i - 1] == w2[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

        index = L[m][n]

        lcs_algo = [""] * (index + 1)
        lcs_algo[index] = ""

        i = m
        j = n
        while i > 0 and j > 0:
            if w1[i - 1] == w2[j - 1]:
                lcs_algo[index - 1] = w1[i - 1]
                i -= 1
                j -= 1
                index -= 1
            elif L[i - 1][j] > L[i][j - 1]:
                i -= 1
            else:
                j -= 1

        return len(lcs_algo)

    # Find the word in the list with the highest similarity score
    close_match = max(l, key=lambda t: lcs_similarity(x.lower(), t.lower()))
    return close_match


# %% [code] Cell 57
def cb_similar_movies_recommendation(similarity_mtrx, movies, suggest_n=5):
    """
    Recommends a list of movies similar to the user's favorite movie using content-based filtering.

    :param numpy.ndarray of float similarity_mtrx: 2D matrix representing similarity scores between all movies
    :param pandas.DataFrame movies: DataFrame containing movie data, including titles
    :param int suggest_n: Number of similar movies to recommend (default is 5)
    :return: Tuple containing the list of recommended movie titles and the movie closest to the user's input
    """
    # CB Movie Recommendation System

    # Prompting the user to enter their favorite movie name
    title = input("Enter your favorite movie name: ")

    # Creating a list with all the movie titles given in the dataset
    titles = movies["title"].tolist()

    # Finding the close match for the movie name given by the user

    movie = most_similar_word(title, titles)
    print(f"The closest match in the database to your favorite movie is: {movie}")

    # Finding the index of the movie with the closest match title
    ith_movie = movies[movies["title"] == movie].index.values[0]

    # Getting a list of similar movies based on similarity scores
    sim_scores = list(enumerate(similarity_mtrx[ith_movie]))

    # Sorting the movies based on their similarity score
    sim_movies = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top similar movies
    print("\nWe recommend you the following movies:")
    to_suggest = []
    for i in range(1, suggest_n + 1):
        ith_movie = sim_movies[i][0]
        sim_movie = movies.iloc[ith_movie]["title"]
        to_suggest.append(sim_movie)
        print(f"{i}. {sim_movie}")

    return to_suggest, movie


# %% [code] Cell 58
recommended_similar_movies, from_movie = cb_similar_movies_recommendation(
    similarity_mtrx=cosine_sim, movies=data_movies, suggest_n=10
)

# %% [markdown] Cell 59
# # Hybrid Recommender - Two Recommender Ensemble

# %% [markdown] Cell 60
# ## CF recommender extension for hybrid systems


# %% [code] Cell 61
def recommend_movies(user, model, users_watched_list, suggest_n=10):
    """
    Recommends movies to a user based on collaborative filtering (CF) predictions from a model.

    :param int user: The UserID of the user for whom the movie recommendations are to be generated.
    :param surprise.SVD model: Trained recommendation model (e.g., using collaborative filtering) that predicts user ratings for movies.
    :param pandas.DataFrame users_watched_list: DataFrame containing user-movie interactions, including 'UserID', 'id' (movie ID), and 'title'.
    :param int suggest_n: Number of top recommendations to return (default is 10).
    :return: A list of recommended movie titles for the user.
    """
    # Get a list of all the movies the user has not rated
    movies = users_watched_list["id"].unique()
    watched = users_watched_list[users_watched_list["UserID"] == user]["id"].unique()
    to_watch = list(set(movies) - set(watched))

    # Predict the ratings for the new movies
    estimated_ratings = [model.predict(user, movie) for movie in to_watch]

    # Sort the predictions by estimated rating
    estimated_ratings.sort(key=lambda i: i.est, reverse=True)

    # Get the top 10 recommendations
    top_recommendations = [
        (
            p.iid,
            users_watched_list[users_watched_list["id"] == p.iid]["title"].values[0],
            p.est,
        )
        for p in estimated_ratings[:suggest_n]
    ][:suggest_n]

    return top_recommendations


# %% [code] Cell 62
user_rated_movies_list_print(39, movie_user_data)
print("\nFor this user we recommend:")

recommend_movies(user=39, model=algo, users_watched_list=movie_user_data, suggest_n=10)

# %% [markdown] Cell 63
# ## CB recommender extension for hybrid systems

# %% [markdown] Cell 64
# ### Auxiliary functions for CB
#

# %% [code] Cell 65
data_movies.info()


# %% [code] Cell 66
# Function to get movie recommendations based on a single movie
def get_movie_recommendations_by_movie(movie, similarity_mtrx, movies, suggest_n=10):

    movies_idx = movies[movies["id"] == movie].index[0]
    # Getting a list of similar movies based on similarity scores
    similarity_score = similarity_mtrx[movies_idx]

    # Sorting the movies based on their similarity score
    sorted_similar_movies = np.argsort(similarity_score)[::-1]

    # Get top similar movies (ignoring the first one since it's the input movie)
    recommended_movies = [
        (
            movies.iloc[movie_index]["id"],
            movies.iloc[movie_index]["title"],
            similarity_score[movie_index],
        )
        for movie_index in sorted_similar_movies[1 : suggest_n + 1]
    ]

    return recommended_movies


# %% [code] Cell 67
interstellar_id = data_movies.loc[data_movies["title"] == "Interstellar"]["id"].values[
    0
]
get_movie_recommendations_by_movie(interstellar_id, cosine_sim, data_movies)


# %% [code] Cell 68
def user_rated_movies_list(user, users_watched_list):
    """
    Retrieves a list of movies rated by a specific user along with their corresponding ratings.

    :param int user: The UserID of the user whose rated movies are to be retrieved
    :param pandas.DataFrame users_watched_list: DataFrame containing user ratings and watched movies, including 'UserID', 'id' (movie ID), 'title', and 'Ratings'
    :return: A tuple containing two lists - the first list consists of movie titles, and the second list contains the corresponding ratings
    """
    # Get the movies rated by the user
    movies = []

    # Get the unique movie IDs watched by the user
    watched = users_watched_list[users_watched_list["UserID"] == user]["id"].unique()

    # Retrieve movie titles and ratings for each movie the user has watched
    movies = [
        (
            users_watched_list[users_watched_list["id"] == movie]["id"].values[0],
            users_watched_list[users_watched_list["id"] == movie]["title"].values[0],
            users_watched_list[users_watched_list["id"] == movie]["Ratings"].values[0],
        )
        for movie in watched
    ]

    return movies


# %% [markdown] Cell 69
# ### CB user-based recommender system


# %% [code] Cell 70
def cb_recommendation_user_based(
    movie_ids, ratings, similarity_mtrx, movies, suggest_n=10, correcting_factor=0.1
):

    from collections import defaultdict

    # Aggregate recommendations for all rated movies by the user
    movie_recommendations = defaultdict(
        list
    )  # Dictionary to store all movie recommendations and scores
    weighted_scores = defaultdict(
        float
    )  # To keep track of weighted scores of recommendations
    scores = {}

    # Loop over each rated movie and its rating
    for movie_id, rating in zip(movie_ids, ratings):
        recommendations = get_movie_recommendations_by_movie(
            movie_id, similarity_mtrx, movies, suggest_n
        )  # Get top recommendations for the movie

        # Add recommendations to dictionary, adjusting score by user's rating
        # recommendations has the structure: (movie_id, movie_title, similarity_score)
        for rec_movie, _, score in recommendations:
            if rec_movie not in movie_ids:
                movie_recommendations[rec_movie].append((score, rating))

                # Count how many times the same movie is recommended
                if rec_movie not in scores:
                    scores[rec_movie] = score
                    weighted_scores[rec_movie] = score * rating
                else:
                    scores[rec_movie] += score
                    weighted_scores[rec_movie] += score * rating

    # Compute final weighted score for each movie
    final_recommendations = []

    for movie, _ in scores.items():
        avg_rating_weight = weighted_scores[movie] / (scores[movie] + correcting_factor)
        final_recommendations.append(
            (
                movie,
                movies.loc[movies["id"] == movie]["title"].values[0],
                avg_rating_weight,
            )
        )

    # Sort the recommendations based on the final weighted score
    final_recommendations.sort(key=lambda x: x[2], reverse=True)

    # Extract the top 'k' recommendations
    top_recommended_movies = final_recommendations[:suggest_n]

    return top_recommended_movies


# %% [code] Cell 71
user_rated_movies_list_print(39, movie_user_data)
print("\nFor this user we recommend:")

movies = user_rated_movies_list(39, movie_user_data)
movie_ids = [id for id, _, _ in movies]
ratings = [rating for _, _, rating in movies]

cb_recommendation_user_based(
    movie_ids, ratings, cosine_sim, data_movies, suggest_n=10, correcting_factor=0.2
)


# %% [code] Cell 72
class ContentBasedRecommender:

    def __init__(
        self, flattened_URM, similarity_matrix, data_movies, correcting_factor=None
    ):
        self.movie_user_data = movie_user_data
        self.similarity_matrix = similarity_matrix
        self.flattened_URM = flattened_URM
        self.data_movies = data_movies
        if correcting_factor is None:
            self.correcting_factor = 0.2
        else:
            self.correcting_factor = correcting_factor

    def recommend(self, user_id, k=5):
        # Get the movies rated by the user
        movie_user_indx = self.flattened_URM[
            self.flattened_URM["UserID"] == user_id
        ].index

        user_rated_movies_ids = self.flattened_URM.loc[movie_user_indx]["id"]
        user_rated_movies_rating = self.flattened_URM.loc[movie_user_indx]["Ratings"]

        # Get the recommended movies
        user_movies = cb_recommendation_user_based(
            user_rated_movies_ids,
            user_rated_movies_rating,
            self.similarity_matrix,
            self.data_movies,
            suggest_n=k,
            correcting_factor=self.correcting_factor,
        )

        movies_ids = [id for id, _, _ in user_movies]

        return movies_ids


# %% [code] Cell 73
cb_recommender = ContentBasedRecommender(train_set, cosine_sim, data_movies)

# %% [code] Cell 74
p5 = precision_at_k(
    validation_set, cb_recommender, k=5, relevance_threshold=5, users_to_monitor=100
)

print(f"\nPrecision @ 5: {p5:.4f}")

# %% [code] Cell 75
p10 = precision_at_k(
    validation_set, cb_recommender, k=10, relevance_threshold=5, users_to_monitor=100
)

print(f"\nPrecision @ 10: {p10:.4f}")

# %% [markdown] Cell 76
# ### CB function with a filtering on best rating


# %% [code] Cell 77
def cb_recommendation_user_based_rating_filtering(
    movie_ids, ratings, similarity_mtrx, movies, suggest_n=10
):
    from collections import defaultdict

    # Aggregate recommendations for all rated movies by the user
    final_scores = defaultdict(
        float
    )  # To keep track of weighted scores of recommendations
    avg_final_score = defaultdict(float)
    repeated_counts = {}

    # Loop over each rated movie and its rating
    for idx, (movie, rating) in enumerate(zip(movie_ids, ratings)):
        if rating > 7:
            recommendations = get_movie_recommendations_by_movie(
                movie=movie,
                similarity_mtrx=similarity_mtrx,
                movies=movies,
                suggest_n=suggest_n * 2,  # TODO: why *2?
            )

            for rec_movie, _, score in recommendations:
                if rec_movie not in movie_ids:
                    # Compute the average weighted score iteratively
                    if rec_movie not in repeated_counts:
                        repeated_counts[rec_movie] = 1
                        final_scores[rec_movie] = score
                    else:
                        repeated_counts[rec_movie] += 1
                        final_scores[rec_movie] += score

    # Compute final weighted score for each movie
    final_recommendations = []

    for movie, _ in repeated_counts.items():
        avg_final_score[movie] = final_scores[movie] / repeated_counts[movie]
        final_recommendations.append(
            (
                movie,
                movies[movies.id == movie]["title"].values[0],
                avg_final_score[movie],
            )
        )

    # Sort the recommendations based on the final weighted score
    sorted_recommendations = sorted(
        final_recommendations, key=lambda x: x[2], reverse=True
    )

    # Extract the top 'k' recommendations
    top_recommended_movies = sorted_recommendations[:suggest_n]

    return top_recommended_movies


# %% [code] Cell 78
user_rated_movies_list_print(39, movie_user_data)
rated_movies = user_rated_movies_list(39, movie_user_data)
print("\nFor this user we recommend:")

rated_movies_ids = [id for id, _, _ in rated_movies]
rated_movies_ratings = [rating for _, _, rating in rated_movies]

cb_recommendation_user_based_rating_filtering(
    rated_movies_ids,
    rated_movies_ratings,
    similarity_mtrx=cosine_sim,
    movies=data_movies,
    suggest_n=10,
)


# %% [code] Cell 79
class ContentBasedWithFilteringRecommender:

    def __init__(self, flattened_URM, similarity_matrix, data_movies):
        self.movie_user_data = movie_user_data
        self.similarity_matrix = similarity_matrix
        self.flattened_URM = flattened_URM
        self.data_movies = data_movies

    def recommend(self, user_id, k=5):
        # Get the movies rated by the user
        movie_user_indx = self.flattened_URM[
            self.flattened_URM["UserID"] == user_id
        ].index

        user_rated_movies_ids = self.flattened_URM.loc[movie_user_indx]["id"]
        user_rated_movies_rating = self.flattened_URM.loc[movie_user_indx]["Ratings"]

        # Get the recommended movies
        user_movies = cb_recommendation_user_based_rating_filtering(
            user_rated_movies_ids,
            user_rated_movies_rating,
            self.similarity_matrix,
            self.data_movies,
            suggest_n=k,
        )

        movies_ids = [id for id, _, _ in user_movies]

        return movies_ids


# %% [code] Cell 80
cb_w_filter_recommender = ContentBasedWithFilteringRecommender(
    train_set, cosine_sim, data_movies
)

# %% [code] Cell 81
cb_w_filter_recommender.recommend(39, 5)

# %% [code] Cell 82
p5 = precision_at_k(
    validation_set,
    cb_w_filter_recommender,
    k=5,
    relevance_threshold=5,
    users_to_monitor=100,
)

print(f"\nPrecision @ 5: {p5:.4f}")

# %% [markdown] Cell 83
# ## Mixed Hybrid Recommender System


# %% [code] Cell 84
def ensemble_recommendation_intersection_based(
    user,
    model,
    flattened_URM,
    similarity_mtrx,
    movies,
    users_watched_list,
    suggest_n=10,
):
    """
    Recommends movies to a user by combining collaborative filtering (CF) and content-based (CB) recommendations.
    The final list of recommendations is based on the intersection of movies recommended by both methods,
    or a union of the two if no common movies are found.

    :param int user: The UserID of the user for whom movie recommendations are being generated.
    :param surprise.SVD model: Trained CF recommendation model used to predict ratings for movies.
    :param numpy.ndarray of float similarity_mtrx: 2D matrix representing similarity scores between all movies.
    :param pandas.DataFrame movies: DataFrame containing movie data, including titles.
    :param pandas.DataFrame users_watched_list: DataFrame containing user-movie interactions, including 'UserID', 'id' (movie ID), and 'title'.
    :param int suggest_n: Number of top recommendations to return (default is 10).

    :return: A tuple with three lists: 1. A list of movies recommended based on the intersection or union of CF and CB recommendations. 2.
             A List of movie titles recommended by the content-based filtering system. 3. A list of movie titles recommended by the collaborative filtering system.
    """
    cf_recommended_movies = recommend_movies(
        user=user,
        model=model,
        users_watched_list=users_watched_list,
        suggest_n=suggest_n,
    )
    # Extract only the ids
    cf_suggestions = [id for id, _, _ in cf_recommended_movies]

    # Get the movies rated by the user
    rated_movie_user_indx = flattened_URM[flattened_URM["UserID"] == user].index
    rated_movies_ids = flattened_URM.loc[rated_movie_user_indx]["id"]
    rated_movies_ratings = flattened_URM.loc[rated_movie_user_indx]["Ratings"]

    cb_recommended_movies = cb_recommendation_user_based(
        rated_movies_ids,
        rated_movies_ratings,
        similarity_mtrx=similarity_mtrx,
        movies=movies,
        suggest_n=suggest_n,
    )
    # Extract only ids
    cb_suggestions = [id for id, _, _ in cb_recommended_movies]

    common_movies = list(set(cf_suggestions) & set(cb_suggestions))
    if len(common_movies) == 0:
        common_movies = list(set(cf_suggestions + cb_suggestions))
        # Order the result by similarity score
        common_movies = sorted(
            common_movies,
            key=lambda t: get_movie_recommendations_by_movie(
                movie=t, similarity_mtrx=similarity_mtrx, movies=movies, suggest_n=1
            )[0][2],
            reverse=True,
        )[:suggest_n]
    else:
        if len(common_movies) < suggest_n:
            # Add to the result the union of the two results sorted by similarity
            result_union = list(set(cf_suggestions + cb_suggestions))
            # Order the result by similarity score
            result_union = sorted(
                result_union,
                key=lambda t: get_movie_recommendations_by_movie(
                    movie=t, similarity_mtrx=similarity_mtrx, movies=movies, suggest_n=1
                )[0][1],
                reverse=True,
            )
            # If in result_union we have values that are also in result we remove them from result_union
            result_union = [item for item in result_union if item not in common_movies]
            common_movies = (
                common_movies + result_union[: suggest_n - len(common_movies)]
            )

    return common_movies, cb_suggestions, cf_suggestions


# %% [code] Cell 85
# Use the random function to run the recommendations for another user id, instead of the user 39 used for our study case and qualitative analysis
# user_id = np.random.choice(movie_user_data['UserID'].unique())
user_id = 39
result, cb, cf = ensemble_recommendation_intersection_based(
    user=user_id,
    model=algo,
    flattened_URM=flattened_URM,
    similarity_mtrx=cosine_sim,
    movies=data_movies,
    users_watched_list=movie_user_data,
    suggest_n=10,
)

result_titles = [
    data_movies[data_movies["id"] == id]["title"].values[0] for id in result
]
cb_titles = [data_movies[data_movies["id"] == id]["title"].values[0] for id in cb]
cf_titles = [data_movies[data_movies["id"] == id]["title"].values[0] for id in cf]

user_rated_movies_list_print(user=user_id, users_watched_list=movie_user_data)
print("\nWe recommend:")
print("\n".join(f"{i + 1}.\t{title}" for i, title in enumerate(result_titles)))
print("\n")
print("Given that the two recommender systems recommended:")
print("CB:")
print("\n".join(f"{i + 1}.\t{title}" for i, title in enumerate(cb_titles)))
print("\nCF:")
print("\n".join(f"{i + 1}.\t{title}" for i, title in enumerate(cf_titles)))


# %% [code] Cell 86
class EnsembleRecommender:

    def __init__(
        self, algo, flattened_URM, similarity_mtrx, data_movies, users_watched_list
    ):
        self.algo = algo
        self.flattened_URM = flattened_URM
        self.similarity_mtrx = similarity_mtrx
        self.data_movies = data_movies
        self.users_watched_list = users_watched_list

    def recommend(self, user_id, k=5):
        # Get the movies rated by the user
        movie_user_indx = self.flattened_URM[
            self.flattened_URM["UserID"] == user_id
        ].index

        user_rated_movies_ids = self.flattened_URM.loc[movie_user_indx]["id"]
        user_rated_movies_rating = self.flattened_URM.loc[movie_user_indx]["Ratings"]

        # Get the recommended movies
        result, _, _ = ensemble_recommendation_intersection_based(
            user=user_id,
            model=self.algo,
            flattened_URM=self.flattened_URM,
            similarity_mtrx=self.similarity_mtrx,
            movies=self.data_movies,
            users_watched_list=self.users_watched_list,
            suggest_n=k,
        )

        return result


# %% [code] Cell 87
ensemble_recommender = EnsembleRecommender(
    algo, flattened_URM, cosine_sim, data_movies, movie_user_data
)

p5 = precision_at_k(
    validation_set,
    ensemble_recommender,
    k=5,
    relevance_threshold=5,
    users_to_monitor=100,
)

print(f"\nPrecision @ 5: {p5:.4f}")

# %% [markdown] Cell 88
# ## Meta-level Hybrid Recommendation System


# %% [code] Cell 89
def ensemble_recommendation_meta_level(
    user, model, similarity_mtrx, users_watched_list, movies, suggest_n=10
):
    """
    Provides movie recommendations to a user using a meta-level ensemble approach that combines collaborative filtering
    (CF) recommendations with content-based (CB) recommendations.

    :param int user: The UserID of the user for whom recommendations are to be generated.
    :param surprise.SVD model: Trained collaborative filtering recommendation model used to predict ratings for movies.
    :param numpy.ndarray of float similarity_mtrx: 2D matrix representing similarity scores between all movies for content-based recommendations.
    :param pandas.DataFrame users_watched_list: DataFrame containing user-movie interactions, including 'UserID', 'id' (movie ID), and 'title'.
    :param pandas.DataFrame movies: DataFrame containing movie information, including titles.
    :param int suggest_n: Number of top recommendations to return (default is 10).

    :return: A list of tuples where each tuple contains a recommended movie title and its associated score, sorted by score in descending order.
    """
    ids = []
    titles = []
    scores = []

    cf_recs = recommend_movies(
        user=user,
        model=model,
        users_watched_list=users_watched_list,
        suggest_n=suggest_n,
    )

    for id, _, _ in cf_recs:
        cb_recs = get_movie_recommendations_by_movie(
            movie=id,
            similarity_mtrx=similarity_mtrx,
            movies=movies,
            suggest_n=suggest_n,
        )

        for cb_id, title, score in cb_recs:
            if cb_id not in ids:
                ids.append(cb_id)
                scores.append(score)
                titles.append(title)

    # Zip together scores and results
    recommendations = list(zip(ids, titles, scores))
    # Sort results_scores by score in descending order
    recommendations = sorted(recommendations, key=lambda x: x[2], reverse=True)

    return recommendations[:suggest_n]


# %% [code] Cell 90
user_rated_movies_list_print(user=user_id, users_watched_list=movie_user_data)
print("\nWe recommend:")

ensemble_recommenddation_meta_level(
    user=user_id,
    model=algo,
    similarity_mtrx=cosine_sim,
    users_watched_list=movie_user_data,
    movies=data_movies,
    suggest_n=10,
)


# %% [code] Cell 91
class MetaEnsembleRecommender:

    def __init__(self, algo, similarity_mtrx, data_movies, users_watched_list):
        self.algo = algo
        self.flattened_URM = flattened_URM
        self.similarity_mtrx = similarity_mtrx
        self.data_movies = data_movies
        self.users_watched_list = users_watched_list

    def recommend(self, user_id, k=5):
        # Get the movies rated by the user
        movie_user_indx = self.flattened_URM[
            self.flattened_URM["UserID"] == user_id
        ].index

        user_rated_movies_ids = self.flattened_URM.loc[movie_user_indx]["id"]
        user_rated_movies_rating = self.flattened_URM.loc[movie_user_indx]["Ratings"]

        # Get the recommended movies
        result = ensemble_recommendation_meta_level(
            user=user_id,
            model=self.algo,
            similarity_mtrx=self.similarity_mtrx,
            movies=self.data_movies,
            users_watched_list=self.users_watched_list,
            suggest_n=k,
        )

        ids = [id for id, _, _ in result]
        return ids


# %% [code] Cell 92
meta_ensemble_recommender = MetaEnsembleRecommender(
    algo, cosine_sim, data_movies, movie_user_data
)

# %% [code] Cell 93
p5 = precision_at_k(
    validation_set,
    meta_ensemble_recommender,
    k=5,
    relevance_threshold=5,
    users_to_monitor=100,
)

print(f"\nPrecision @ 5: {p5:.4f}")

# %% [markdown] Cell 94
# ## Hybrid recommender - changing CB adding overview information

# %% [markdown] Cell 95
# Extract the overview from the dataset and create embeddings of the overview, to use it as a feature for the movie similarity. Let's see if this creates a more accurate model.

# %% [code] Cell 96
# We can create a single feature vector containing all features
selected_features = [
    "Genre",
    "production_companies",
    "actors",
    "directors",
    "writers",
    "production_countries",
    "original_language",
]

# Handling missing values and converting to strings
data_movies[selected_features] = (
    data_movies[selected_features]
    .fillna("")
    .applymap(lambda x: " ".join(x if isinstance(x, list) else [str(x)]))
)

# Combining selected features into a single feature vector
# combined_features keeps a comprehensive representation of each movie's characteristics
# expand the features that are lists in string
combined_features_ovw = data_movies[selected_features].agg(" ".join, axis=1)

# Add movies_attributes['overview'] to combined_features
movie_attributes["overview"] = movie_attributes["overview"].fillna("")
movie_attributes["overview"] = movie_attributes["overview"].apply(
    lambda x: x if isinstance(x, list) else [str(x)]
)
movie_attributes["overview"] = movie_attributes["overview"].apply(lambda x: " ".join(x))

combined_features_ovw = combined_features_ovw + " " + movie_attributes["overview"]
# Drop rows with NaN values
combined_features_ovw = combined_features_ovw.dropna()
# Convert to lowercase
combined_features_ovw = combined_features_ovw.str.lower()

# %% [code] Cell 97
combined_features_ovw[0]

# %% [code] Cell 98
# Converting Text Data to Feature Vectors
# Converting the combined features into feature vectors
feature_vectors_ovw = vectorizer.fit_transform(combined_features_ovw)

feature_vectors_ovw.shape

# %% [code] Cell 99
# Create the similarity matrix
# Getting the similarity scores using cosine similarity
cosine_sim_ovw = cosine_similarity(feature_vectors_ovw)

recommended_similar_movies_ovw, from_movie_ovw = cb_similar_movies_recommendation(
    similarity_mtrx=cosine_sim_ovw, movies=data_movies, suggest_n=10
)

# %% [code] Cell 100
user_rated_movies_list_print(user=user_id, users_watched_list=movie_user_data)
print("\nWe recommend:")

ensemble_recommendation_meta_level(
    user=user_id,
    model=algo,
    similarity_mtrx=cosine_sim_ovw,
    users_watched_list=movie_user_data,
    movies=data_movies,
    suggest_n=10,
)

# %% [code] Cell 101
result, cb, cf = ensemble_recommendation_intersection_based(
    user=user_id,
    model=algo,
    flattened_URM=flattened_URM,
    similarity_mtrx=cosine_sim_ovw,
    movies=data_movies,
    users_watched_list=movie_user_data,
    suggest_n=10,
)

result_titles = [
    data_movies[data_movies["id"] == id]["title"].values[0] for id in result
]
cb_titles = [data_movies[data_movies["id"] == id]["title"].values[0] for id in cb]
cf_titles = [data_movies[data_movies["id"] == id]["title"].values[0] for id in cf]

user_rated_movies_list_print(user=user_id, users_watched_list=movie_user_data)
print("\nWe recommend:")
print("\n".join(f"{i + 1}.\t{title}" for i, title in enumerate(result_titles)))
print("\n")
print("Given that the two recommender systems recommended:")
print("CB:")
print("\n".join(f"{i + 1}.\t{title}" for i, title in enumerate(cb_titles)))
print("\nCF:")
print("\n".join(f"{i + 1}.\t{title}" for i, title in enumerate(cf_titles)))

# %% [code] Cell 102
ensemble_recommender = EnsembleRecommender(
    algo, flattened_URM, cosine_sim_ovw, data_movies, movie_user_data
)

p5 = precision_at_k(
    validation_set,
    ensemble_recommender,
    k=5,
    relevance_threshold=5,
    users_to_monitor=100,
)

print(f"\nPrecision @ 5: {p5:.4f}")

# %% [markdown] Cell 103
# ## Get the ground-truth to evaluate the model

# %% [markdown] Cell 104
# ### Groundtruth for all movies


# %% [code] Cell 105
# Download ground-truth - the movies that should actually be recommended given a certain movie, according to TMDb
def download_ground_truth(key, users_watched_list):
    import json
    import os.path
    import time

    if not os.path.exists(ground_truth_path):
        wait_time_s = 1
        all_data = {}

        # Get from movie_user_data the id of the movies, and create a set of movie ids, without repetitions
        movie_list = list(set(users_watched_list["id"]))

        for i in range(0, len(movie_list)):
            ith_movie = movie_list[i]

            # Set headers
            headers = {"accept": "application/json"}

            # Build URL
            url = f"https://api.themoviedb.org/3/movie/{ith_movie}/recommendations?api_key={key}"

            # Send API Call
            time.sleep(0.01)
            response = requests.get(url, headers)

            if response.status_code == 200:
                # store the Json data in a list:
                all_data[ith_movie] = response.json()
                print(f"Got: {ith_movie}")
            else:
                print(f"Error: {response.status_code}")

            time.sleep(wait_time_s)

        # write the list to file
        with open(ground_truth_path, "w") as f_out:
            json.dump(all_data, f_out, indent=4)
        print("Done")
    else:
        print("File already exists")


# %% [code] Cell 106
download_ground_truth(api_key, users_watched_list=movie_user_data)

# %% [code] Cell 107
ground_truth = pd.read_json(ground_truth_path).T

# %% [code] Cell 108
ground_truth_id = ground_truth.index
# Extract the values from results
ground_truth = pd.json_normalize(ground_truth["results"])
ground_truth["id"] = ground_truth_id


# %% [code] Cell 109
def get_ground_truth(attribute, expected_rec):
    """
    Extracts and compiles the ground truth attribute (e.g., 'id') from a DataFrame of expected recommendations.

    :param str attribute: The attribute to be extracted from each dictionary in the columns (e.g., 'id').
    :param pandas.DataFrame expected_rec: A DataFrame containing expected recommendations, where each column may contain dictionaries with the desired attribute.

    :return: A Series with a compressed list of the extracted attribute (e.g., 'id') for each row.
    """
    # Initialize an empty DataFrame to store the extracted attribute
    expected_rec_attribute = pd.DataFrame()

    # Iterate through each column, except 'id', to extract the desired attribute
    for i in range(0, len(expected_rec.drop("id", axis=1).columns)):
        # Check if the value is a dictionary and extract the attribute (e.g., 'id')
        expected_rec_attribute[i] = expected_rec[i].apply(
            lambda x: str(x[attribute]) if isinstance(x, dict) else None
        )

    # Compress the attribute columns into a single column by combining non-null values into lists
    expected_rec_attribute[attribute] = expected_rec_attribute.apply(
        lambda x: list(x.dropna()), axis=1
    )

    # Drop all the other columns and keep only the compressed attribute column
    expected_rec_attribute = expected_rec_attribute[attribute]

    # Return the final Series containing the ground truth data
    return expected_rec_attribute


# %% [code] Cell 110
ground_truth_id = get_ground_truth("id", ground_truth)
ground_truth_title = get_ground_truth("title", ground_truth)
ground_truth_release_date = get_ground_truth("release_date", ground_truth)

ground_truth_rec = pd.concat(
    [ground_truth_id, ground_truth_title, ground_truth_release_date], axis=1
)
ground_truth_rec["movie_id"] = ground_truth["id"]

ground_truth_rec

# %% [markdown] Cell 111
# ### Evaluate the models based on the ground-truth

# %% [code] Cell 112
# Print the results of the CB recommender system for the chosen movie
print(f"For {from_movie}\n\nWe recommend:")

for i in range(len(recommended_similar_movies)):
    print(f"{i + 1}. {recommended_similar_movies[i]}")

# %% [code] Cell 113
movie_id = data_movies[data_movies["title"] == from_movie]["id"].values[0]

# From ground_truth_rec get the row with id as movie_id
truth = ground_truth_rec[ground_truth_rec["movie_id"] == movie_id]

# %% [code] Cell 114
# Extract the truth['title'] values as a NumPy array and flatten it
truth_list = truth["title"].values.flatten()

# Extract only the values inside the list
truth_list = [item for sublist in truth_list for item in sublist]

# Remove the appendices from each title of the truth_list
truth_list = [title.split(" (")[0] for title in truth_list]

truth_list

# %% [code] Cell 115
# Find how many matches we have between recommended_similar_movies and the elements of the list of truth['title']
matches = [title for title in recommended_similar_movies if title in truth_list]
count = len(matches)

print(f"Our model recommended {count} movies from the ground-truth.")
print(matches)  # With Interstellar we have a match
