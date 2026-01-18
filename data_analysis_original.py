# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [markdown] Cell 1
# # Connect to Drive

# %% [code] Cell 2
from google.colab import drive
drive.mount('/content/drive')

# %% [code] Cell 3
import os

work_directory = '/content/drive/MyDrive/Colab Notebooks/NAML/NAML_Project/'
work_directory = '/content/drive/MyDrive/Colab Notebooks/NAML_Labs/NAML_Project'
data_directory = os.path.join(work_directory, 'Data')

os.chdir(work_directory)
os.getcwd()

# %% [code] Cell 4
# Load the .env file from Google Drive
import os
from dotenv import load_dotenv
load_dotenv(os.path.join('/content/drive/MyDrive', '.env'))

api_key = os.getenv('TMDB_API_KEY')
tmdbs_auth_token = os.getenv('TMDB_AUTH_TOKEN')

# %% [markdown] Cell 5
# # Load Packages

# %% [code] Cell 6
import json
import os.path
import time
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import seaborn as sns
from matplotlib import pyplot as plt

# %% [code] Cell 7
url = "https://api.themoviedb.org/3/authentication"

headers = {
    "accept": "application/json",
    "Authorization": tmdbs_auth_token
}

# %% [markdown] Cell 8
# The following files paths will be used across the file to save and load data. In particular, the data we had a priori were `movies.dat`, `ratings.dat`, and `users.dat`. However, the users dataset was not used because we acknowledged its limited usability in scenarios where the sentiment analysis step was already solved and given to us.
# 
# Sentiment analysis was described in the paper as the first step of the final model, but we already had its information in the `ratings.dat` file.

# %% [code] Cell 9
movie_credits_path = work_directory + 'Data/movie_credits.json'
movies_path = work_directory + 'Data/movies.dat'
movie_tmdb_attributes_path = work_directory + 'Data/movie_tmdb_attributes.json'
ratings_path = work_directory + 'Data/ratings.dat'
tmdb_by_title_path = work_directory + 'Data/tmdb_by_title.json'
tmdb_imdb_association_path = work_directory + 'Data/tmdb_imdb_association.json'
tmdb_movies_path = work_directory + 'Data/tmdb_movies.json'
imdb_movies_path = work_directory + 'Data/tmdb_movies_MovieID.json'

# %% [code] Cell 10
def format_input(voices, values):
    len_str = max(map(len, voices)) + max(map(len, values))
    for voice, value in zip(voices, values):
        print(f"·\t{voice}: {' ' * (len_str - len(voice) - len(value))}{value}")

# %% [code] Cell 11
def download_data_by_movie_title(file_path, df):
    if os.path.exists(file_path):
        print('File already exists')
        return

    all_data, rate_limit, wait_time_s = [], 50, 1

    for i in range(len(df)):
        for _ in range(rate_limit):
            movie_name = df['Title'].iloc[i]
            url = f'https://api.themoviedb.org/3/search/movie?query={movie_name}'
            time.sleep(0.01)
            response = requests.get(url, headers)

            if response.status_code == 200:
                all_data.append(response.json())
                print(f'Got: {movie_name}')
            else:
                print(f'Error: {response.status_code}')

        time.sleep(wait_time_s)

    with open(file_path, 'w') as f_out:
        json.dump(all_data, f_out, indent=4)
    print('Done')

# %% [code] Cell 12
def download_data_by_movie_id(file_path, df, key):
    if os.path.exists(file_path):
        print('File already exists')
        return

    all_data, wait_time_s = [], 1

    for movie_id in df['id']:
        url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={key}'
        time.sleep(0.01)
        response = requests.get(url, headers)

        if response.status_code == 200:
            all_data.append(response.json())
            print(f'Got: {movie_id}')
        else:
            print(f'Error: {response.status_code}')

        time.sleep(wait_time_s)

    with open(file_path, 'w') as f_out:
        json.dump(all_data, f_out, indent=4)
    print('Done')

# %% [code] Cell 13
def download_credits(file_path, df, key):
    if os.path.exists(file_path):
        print('File already exists')
        return

    all_data, wait_time_s = [], 1

    for movie_id in df['id']:
        url = f'https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={key}'
        time.sleep(0.01)
        response = requests.get(url, headers)

        if response.status_code == 200:
            all_data.append(response.json())
            print(f'Got: {movie_id}')
        else:
            print(f'Error: {response.status_code}')

        time.sleep(wait_time_s)

    with open('movie_credits.json', 'w') as f_out:
        json.dump(all_data, f_out, indent=4)
    print('Done')

# %% [markdown] Cell 14
# ## Movies Dataset

# %% [markdown] Cell 15
# Contains the items (i.e., movies) that were rated in the tweets, together with their genre metadata in the following format: ``movie_id::movie_title (movie_year)::genre|genre|genre``.
# 
# For example: ``0110912::Pulp Fiction (1994)::Crime|Thriller``.

# %% [code] Cell 16
movies_columns = ["MovieID", 'MovieTitle(Year)', 'Genre']
ratings_columns = ["UserID", "MovieID", "Ratings", "RatingTimestamp"]

print("Let's import the Movies dataset...", end=' ')
data_movies = pd.read_csv(movies_path, delimiter='::', names=movies_columns, engine='python')
print("Done")

len_movies = len(data_movies)

# %% [code] Cell 17
data_movies.info()
data_movies.head()

# %% [markdown] Cell 18
# The `movies.dat` dataset contains 37.342 movies, each one characterized by its title, year and list of genres.
# 
# The title and year attributes are stored in a single feature named `MovieTitle(Year)`, and for this reason we'll have to extract the information and store them in new features. While the attribute `Genre` is not intuitive as it is because it is a string of genres with the char `|` as separator: also this field will be processed.
# 
# We can see how the `Genre` feature is not fully populated because of null values (37.271 non-null values out of 37.342 entries).

# %% [markdown] Cell 19
# ## Ratings Dataset

# %% [markdown] Cell 20
# In this file the extracted ratings are stored in the following format: ``user_id::movie_id::rating::rating_timestamp``.
# 
# For example: ``14927::0110912::9::1375657563``.

# %% [code] Cell 21
print("Let's import the Ratings dataset...", end=' ')
data_ratings = pd.read_csv(ratings_path, delimiter='::', names=ratings_columns, engine='python')
print("Done")

len_ratings = len(data_ratings)

format_input(
    voices=['Movies dataset', 'Ratings dataset'],
    values=[f'{data_movies.shape}', f'{data_ratings.shape}']
)

# %% [code] Cell 22
data_ratings.info()
data_ratings.head()

# %% [markdown] Cell 23
# The `ratings.dat` dataset contains 906.831 ratings of different users about different movies. Each sample is identified by the primary key `(UserID, MovieID)`, because this result comes from the Tweetings DB analysis, so it was a relational table with foreign key the fields `UserID` and `MovieID` in, respectively, the datasets `users.dat` and `movies.dat`.
# 
# Here, we are highly interested in the `Ratings` to create Movie Recommendation Systems.

# %% [markdown] Cell 24
# # Preprocess Movies DB and Analyze data

# %% [markdown] Cell 25
# ## Clean Movies Dataset

# %% [markdown] Cell 26
# Remove exact duplicate entries from the movies dataset using `drop_duplicates()`, which modifies the data in place. Then, calculate and print the percentage of remaining movies relative to the original dataset. Finally, print the updated shape of the dataset and store the new length of the movies dataset in `len_movies`.

# %% [code] Cell 27
print(f"Drop the exact duplicated samples...", end=' ')
data_movies.drop_duplicates(inplace=True, ignore_index=True)
print("Done")

print(f"We kept {data_movies.shape[0] / len_movies * 100:.0f}% of the movies.")
print(f"Movies: {data_movies.shape}")

len_movies = len(data_movies)

# %% [markdown] Cell 28
# This block processes the `Genre` column by splitting the genre strings at each `|` and converting them into lists. It then resets the index of the DataFrame while dropping the old index. Finally, it removes any rows where the `Genre` column contains missing values (`NaN`).

# %% [code] Cell 29
data_movies['Genre'] = data_movies['Genre'].str.split('|')
data_movies.reset_index(drop=True, inplace=True)
data_movies.dropna(subset=['Genre'], inplace=True)

# %% [code] Cell 30
data_movies.info()
data_movies.head()

# %% [markdown] Cell 31
# Extract and count the frequency of each genre from the `Genre` column of the movies dataset using `Counter` to accumulate the genres across all movies.

# %% [code] Cell 32
print("Let's extract the genres...", end=' ')
genres = Counter(
    g for genres in data_movies['Genre'].values for g in genres
).most_common()
print("Done")

print(f"Number of genres: {len(genres)}")
print("The genres with the relative number of movies")
format_input(
    voices=[genre for (genre, _) in genres],
    values=[f"{value}" for (_, value) in genres]
)

genres_df = pd.DataFrame(genres, columns=['Genre', 'Count'])
px.pie(
    values=[value for (_, value) in genres],
    names=[genre for (genre, _) in genres],
    title="Distribution of the Genres"
).show()

# %% [markdown] Cell 33
# This code block filters out movies with undefined or non-list genres by applying a lambda function that keeps only rows where the `Genre` column is a list. It then prints the percentage of the dataset retained after this filtering step. Finally, it displays the new shape of the dataset and updates the `len_movies` variable with the new count of movies.

# %% [code] Cell 34
print("Let's remove the movies with undefined genres...", end=' ')
data_movies = data_movies[data_movies['Genre'].apply(lambda x: isinstance(x, list))]
print("Done")

print(f"We kept {data_movies.shape[0] / len_movies * 100:.0f}% of the dataset.")
print(f"Movies: {data_movies.shape}")

len_movies = len(data_movies)

# %% [code] Cell 35
data_movies.head()

# %% [markdown] Cell 36
# This code filters the dataset to retain only movies whose genres are all among those with at least 20 occurrences, determined by the `popular_genres` list. It applies a lambda function to check that every genre in the `Genre` column belongs to this list. After filtering, it prints the percentage of the dataset kept and updates the shape and length of the remaining movies in the dataset.

# %% [code] Cell 37
print("Let's remove the movies with infrequent genres...", end=' ')
popular_genres = [genre for genre, count in genres if count >= 20]
data_movies = data_movies[data_movies['Genre'].apply(lambda genre: all(g in popular_genres for g in genre))]
print("Done")

print(f"We kept {data_movies.shape[0] / len_movies * 100:.0f}% of the dataset.")
print(f"Movies: {data_movies.shape}")

len_movies = len(data_movies)

# %% [markdown] Cell 38
# Re-extract the genres and their frequencies from the `Genre` column using `Counter`, as done previously. Additionally, it creates a new DataFrame `genres_df` to store the genres and their counts, and displays a pie chart visualizing the distribution of genres across the movies dataset.

# %% [code] Cell 39
print("Let's extract the genres...", end=' ')
genres = Counter(
    g for genres in data_movies['Genre'].values for g in genres
).most_common()
print("Done")

print(f"Number of genres: {len(genres)}")
print("The genres with the relative number of movies")
format_input(
    voices=[genre for (genre, _) in genres],
    values=[f"{value}" for (_, value) in genres]
)

genres_df = pd.DataFrame(genres, columns=['Genre', 'Count'])
px.pie(
    values=[value for (_, value) in genres],
    names=[genre for (genre, _) in genres],
    title="Distribution of the Genres"
).show()

# %% [markdown] Cell 40
# Compute the correlation matrix between movies' genres. First, it flattens the list of genres across all movies and creates a unique list of genres. It then constructs a binary matrix where each column represents a genre, and each movie is marked with `1` for the genres it belongs to. After populating this matrix, it calculates the correlation between genres and visualizes the correlation matrix using a clustered heatmap without row dendrograms.

# %% [code] Cell 41
print("Let's compute the correlation matrix between movies' genres...", end=' ')
# Flatten the list of all genres to create a unique list of genres
all_genres = [genre for sublist in data_movies['Genre'] for genre in sublist]
unique_genres = list(set(all_genres))

# Create a binary matrix for genres
genre_matrix = pd.DataFrame(0, index=data_movies.index, columns=unique_genres)

# Populate the binary matrix
for idx, genres in enumerate(data_movies['Genre']):
    genre_matrix.loc[idx, genres] = 1

# Compute the correlation matrix
correlation_matrix = genre_matrix.corr()
print("Done")

g = sns.clustermap(correlation_matrix, center=0, cmap="vlag")
g.ax_row_dendrogram.remove()

# %% [markdown] Cell 42
# Process the `MovieTitle(Year)` column by extracting the `Title` and `Release_year` features using a regular expression that splits the title and the year. Assign these extracted values into two new columns, `Title` and `Release_year`, while converting `Release_year` to a numeric format.

# %% [code] Cell 43
print("Let's process the 'MovieTitle(Year)' to extract the 'Title' and 'Release_year' features...", end=' ')
data_movies[['Title', 'Release_year']] = data_movies.pop('MovieTitle(Year)').str.extract(r'(.*)\s+\((\d+)\)')
data_movies['Release_year'] = pd.to_numeric(data_movies.pop('Release_year'))
print("Done")

print(f"Movies dataset: {data_movies.shape}")

# %% [code] Cell 44
data_movies.head()

# %% [markdown] Cell 45
# Create a histogram that visualizes the number of movies released per year. The plot offers insights into the release trends of movies over time.

# %% [code] Cell 46
sns.histplot(
    data=data_movies,
    x='Release_year',
    discrete=True,
    element='step'
)
plt.title('Number of Movies Released per Year')
plt.show()

# %% [markdown] Cell 47
# The plot shows the number of movies released per year: there's a general upward tren in movie releases over time, with a dramatic increase starting around 1990. The number of movies released peaks around 2013-2015, reaching over 2.000 movies per year. However, after 2017, there's a steep drop in the number of movies, falling to around 750 by 2020.
# 
# The period 2014 to 2017 represents the peak of movie releases, providing the largest and most diverse sample of films, and they are recent enough to reflect contemporary trend in filmmaking and audience preferences. Also, with more movies released, there was likely more tweet data available for analysis, potentially leading to more robust sentiment analysis results. For these reasons, we chose to keep only the movies released in between 2014 and 2017.

# %% [markdown] Cell 48
# Filter the dataset to retain only the movies released between 2014 and 2017, inclusive.

# %% [code] Cell 49
print("Let's retain only the movies released in between 2014 and 2017...", end=' ')
data_movies = data_movies.query('2014 <= Release_year <= 2017').reset_index(drop=True)
print("Done")

print(f"We kept {data_movies.shape[0] / len_movies * 100:.0f}% of the movies.")
print(f"Movies: {data_movies.shape}")

len_movies = len(data_movies)

# %% [code] Cell 50
data_movies.info()

# %% [markdown] Cell 51
# Generate a histogram of movie releases per year using the `Release_year` column from the filtered dataset. The plot allows for an examination of movie release trends within the specified years.

# %% [code] Cell 52
sns.histplot(
    data=data_movies,
    x='Release_year',
    discrete=True,
    element='step'
)
plt.title('Number of Movies Released per Year')
plt.show()

# %% [markdown] Cell 53
# From the original 37342 entries, we retained only 8143 movies.

# %% [markdown] Cell 54
# First, extract and count the frequency of each genre from the `Genre` column in the dataset using `Counter`. Additionally, create a DataFrame `genres_df` to store the genre counts and visualize the distribution of genres with a pie chart.

# %% [code] Cell 55
print("Let's extract the genres...", end=' ')
genres = Counter(
    g for genres in data_movies['Genre'].values for g in genres
).most_common()
print("Done")

print(f"Number of genres: {len(genres)}")
print("The genres with the relative number of movies")
format_input(
    voices=[genre for (genre, _) in genres],
    values=[f"{value}" for (_, value) in genres]
)

genres_df = pd.DataFrame(genres, columns=['Genre', 'Count'])
px.pie(
    values=[value for (_, value) in genres],
    names=[genre for (genre, _) in genres],
    title="Distribution of the Genres"
).show()

# %% [markdown] Cell 56
# ## Clean Ratings Dataset

# %% [markdown] Cell 57
# Present a comparison of the number of unique movies in two datasets: the Ratings dataset and the Movies dataset. It displays the count of unique `MovieID` values from each dataset. This helps to quickly assess how many distinct movies are present in each dataset.

# %% [code] Cell 58
format_input(
    voices=['Movies in the Ratings dataset', 'Movies in the Movies dataset'],
    values=[f"{data_ratings['MovieID'].nunique()}", f"{data_movies['MovieID'].nunique()}"]
)

# %% [markdown] Cell 59
# The ratio between the two datasets is potentially $\frac{37337}{8138} \approx 21.80\%$ because the missing movies were inside the dropped part of the dataset. This is an issue for us because we have to discard a huge part of the Ratings dataset, which is not good in data scarcity problems. However, we continued because we also wanted to prove the paper thesis, even if potentially we would like to extend our dataset to more ratings about older movies.
# 
# Now we need to remove the ratings associated with movies released outside the year 2014-2017 range.

# %% [markdown] Cell 60
# Filter the `data_ratings` DataFrame to keep only the ratings for movies that are also present in the `data_movies` dataset. Use the `query()` method with the `in` operator to perform this filtering and updates the DataFrame in place.

# %% [code] Cell 61
print("Let's keep only the ratings of movies in the Movies dataset...", end=' ')
data_ratings.query('MovieID in @data_movies.MovieID', inplace=True)
print("Done")

print(f"We kept {data_ratings.shape[0] / len_ratings * 100:.0f}% of the ratings.")

len_ratings = len(data_ratings)

format_input(
    voices=['Movies in the Ratings dataset', 'Number of ratings'],
    values=[f"{data_ratings['MovieID'].nunique()}", f"{len_ratings}"]
)

# %% [code] Cell 62
data_ratings.info()

# %% [markdown] Cell 63
# ### Keep only the ratings of users who have rated at least 20 movies

# %% [markdown] Cell 64
# Filter `data_ratings` to retain only users with at least 20 ratings by grouping the DataFrame by `UserID` and applying a filter condition.

# %% [code] Cell 65
len_users = data_ratings['UserID'].nunique()

print("Let's filter the users to retain only the ones with at least 20 ratings...", end=' ')
data_ratings = data_ratings.groupby('UserID').filter(lambda x: len(x) >= 20)
data_ratings.reset_index(drop=True, inplace=True)
print("Done")

print(f"We kept {data_ratings.shape[0] / len_ratings * 100:.0f}% of the ratings.")
print(f"We kept {data_ratings['UserID'].nunique() / len_users * 100:.0f}% of the users.")

len_ratings = len(data_ratings)
len_users = data_ratings['UserID'].nunique()

format_input(
    voices=['Users in the Ratings dataset', 'Number of ratings'],
    values=[f"{data_ratings['UserID'].nunique()}", f"{len_ratings}"]
)

# %% [code] Cell 66
data_ratings.info()

# %% [markdown] Cell 67
# We can see that from the original 906.831 ratings we have retained 215.618 ratings.

# %% [markdown] Cell 68
# Filter the `data_movies` DataFrame to retain only movies that are present in the `data_ratings` dataset by using the `query()` method with the `in` operator.

# %% [code] Cell 69
print("Let's filter the Movies dataset to retain only the ones in the Ratings dataset...", end=' ')
data_movies.query('MovieID in @data_ratings.MovieID', inplace=True)
data_movies.reset_index(drop=True, inplace=True)
print("Done")

print(f"Movies: {data_movies.shape}")
print(f"We kept {data_movies.shape[0] / len_movies * 100:.0f}% of the movies.")

len_movies = len(data_movies)

# %% [code] Cell 70
data_movies.info()

# %% [markdown] Cell 71
# Extract and count the occurrences of each genre from the `Genre` column in the `data_movies` DataFrame using `Counter`.

# %% [code] Cell 72
print("Let's extract the genres...", end=' ')
genres = Counter(
    g for genres in data_movies['Genre'].values for g in genres
).most_common()
print("Done")

print(f"Number of genres: {len(genres)}")
print("The genres with the relative number of movies")
format_input(
    voices=[genre for (genre, _) in genres],
    values=[f"{value}" for (_, value) in genres]
)

genres_df = pd.DataFrame(genres, columns=['Genre', 'Count'])
px.pie(
    values=[value for (_, value) in genres],
    names=[genre for (genre, _) in genres],
    title="Distribution of the Genres"
).show()

# %% [markdown] Cell 73
# ### Investigate the ratings

# %% [markdown] Cell 74
# Merge the `data_movies` and `data_ratings` datasets on the `MovieID` column, resulting in a combined DataFrame `X`. Then, calculate the distribution of ratings values and visualize this distribution with a pie chart. The chart displays the proportion of each rating value in the merged dataset.

# %% [code] Cell 75
print("Let's merge the Movies and Ratings datasets...", end=' ')
X = data_movies.merge(data_ratings, on="MovieID")
print("Done")

ratings = X['Ratings'].value_counts()
px.pie(X, values=ratings.values, names=ratings.index, title="Distribution of the ratings' values").show()

# %% [markdown] Cell 76
# We can see that ratings 7 and 8 dominate the distribution having a lot of medium-high positive evaluation values.

# %% [markdown] Cell 77
# Identify and display the top `n` movies with the highest number of ratings equal to 10. It uses `query()` to filter the dataset for ratings of 10, then counts the occurrences of each movie title and selects the top `n` based on these counts.

# %% [code] Cell 78
n = 10

print(f"The movies with the highest amount of {n}'s ratings")
format_input(
    voices=[f"{title}" for title in X.query("Ratings == 10")['Title'].value_counts().index[:n]],
    values=[f"{count}" for count in X.query("Ratings == 10")['Title'].value_counts().head(n)]
)

# %% [markdown] Cell 79
# Calculate various statistics on the number of ratings per user in the dataset, including the average, median, mode, maximum, and minimum number of ratings. Compute these statistics using `groupby()` to count ratings per user and then applies functions like `mean()`, `max()`, `min()`, `mode()`, and `median()`.

# %% [code] Cell 80
n_ratings_per_user = X.groupby(['UserID'])['Ratings'].count()
avg_number_ratings = n_ratings_per_user.mean()
max_number_ratings = n_ratings_per_user.max()
min_number_ratings = n_ratings_per_user.min()
mode_number_ratings = n_ratings_per_user.mode()[0]
median_number_ratings = n_ratings_per_user.median()

format_input(
    voices=["Average number of ratings per user", "Median for number of ratings per user",
            "Mode for number of ratings per user", "Max number of ratings per user", "Min number of ratings per user"],
    values=[f"{avg_number_ratings:.2f}", f"{median_number_ratings:.2f}", f"{mode_number_ratings}",
            f"{max_number_ratings}", f"{min_number_ratings}"]
)

# %% [markdown] Cell 81
# Create a histogram to visualize the distribution of the number of ratings per user using Seaborn's `histplot()`. The plot provides insights into how ratings are distributed among users.

# %% [code] Cell 82
sns.histplot(x=n_ratings_per_user, binwidth=20, edgecolor='black')
plt.title('Ratings per user')
plt.show()

# %% [markdown] Cell 83
# Generate a histogram to visualize the distribution of ratings per user. It limits the x-axis range from the minimum number of ratings to 200 because here the ratings counts are dense. The plot offers a view of how user ratings are distributed within the specified range.

# %% [code] Cell 84
sns.histplot(n_ratings_per_user, binwidth=20, binrange=(min_number_ratings, 200), edgecolor='black')
plt.title("Ratings per user")
plt.show()

# %% [markdown] Cell 85
# ## Ratings in Time

# %% [markdown] Cell 86
# Extract information from the Timestamp and visualize the trend of reviews by day.

# %% [markdown] Cell 87
# This code block extracts and processes metadata from the `RatingTimestamp` column by converting Unix timestamps into readable date components and time information. It then creates separate time series for the number of ratings per day for the years 2014 through 2017, and combines these series into one DataFrame. Finally, it visualizes the number of reviews per day for each year using a line plot with Plotly, showing trends in review activity over time.

# %% [code] Cell 88
print("Let's extract the metadata from the 'RatingTimestamp' column...", end=' ')
X[['Date', 'Month', 'Year', 'Time']] = pd.DataFrame(
    [[datetime.utcfromtimestamp(int(ts)).strftime(fmt) for fmt in ('%Y-%m-%d', '%m', '%Y', '%H:%M:%S')]
     for ts in X['RatingTimestamp']]
)
print("Done")

print(f"Dataset: {X.shape}")

print("Let's create the series of ratings in time...", end=' ')
series_rating_2014 = (
    pd.DataFrame(X[X['Release_year'] == 2014].groupby('Date')['UserID'].count())
    .reset_index()
    .sort_values('Date')
)
series_rating_2015 = (
    pd.DataFrame(X[X['Release_year'] == 2015].groupby('Date')['UserID'].count())
    .reset_index()
    .sort_values('Date')
)
series_rating_2016 = (
    pd.DataFrame(X[X['Release_year'] == 2016].groupby('Date')['UserID'].count())
    .reset_index()
    .sort_values('Date')
)
series_rating_2017 = (
    pd.DataFrame(X[X['Release_year'] == 2017].groupby('Date')['UserID'].count())
    .reset_index()
    .sort_values('Date')
)
series_rating_2014['Year'] = 2014
series_rating_2015['Year'] = 2015
series_rating_2016['Year'] = 2016
series_rating_2017['Year'] = 2017
combined_series = pd.concat([series_rating_2014, series_rating_2015, series_rating_2016, series_rating_2017])
print("Done")

px.line(combined_series, x='Date', y='UserID', color='Year',
        title='Number of reviews per day for movies between 2014 and 2017').show()

# %% [markdown] Cell 89
# # Get TMDb data

# %% [markdown] Cell 90
# ### Access credentials to TMDb to make requests

# %% [markdown] Cell 91
# Check access credentials to TheMovieDb by sending a GET request to the specified URL with the provided headers. Then, print the response text, which likely contains information about the success or failure of the request, such as an error message or confirmation of access. This step is crucial for ensuring that the connection to the API is properly authenticated and functional.

# %% [code] Cell 92
print("Let's check the access credentials to TheMovieDb...", end=' ')
response = requests.get(url, headers=headers)
print("Done")

print(response.text)

# %% [markdown] Cell 93
# ### Download Data from TMDb

# %% [markdown] Cell 94
# Call the `download_data_by_movie_title` function, passing `tmdb_by_title_path` and `data_movies` as arguments. This function is responsible for downloading additional data from TheMovieDb API based on movie titles in the `data_movies` DataFrame and saving it to the path specified by `tmdb_by_title_path`. This step is used to enrich the movies dataset with external information from TheMovieDb.

# %% [code] Cell 95
download_data_by_movie_title(tmdb_by_title_path, data_movies)

# %% [markdown] Cell 96
# # Load data and merge

# %% [markdown] Cell 97
# ### Explore the data obtained from TMDb

# %% [markdown] Cell 98
# This step loads the additional movie data obtained from TheMovieDb into the working environment for further analysis or processing.

# %% [code] Cell 99
print("Let's import the raw TMDb dataset...", end=' ')
data_tmdb = pd.read_json(tmdb_by_title_path)
print("Done")

# %% [markdown] Cell 100
# Flatten the nested `results` field from the `data_tmdb` DataFrame, extracting individual movie records into a new DataFrame. This process restructures the raw TMDB dataset for easier analysis and integration with the existing movies' dataset.

# %% [code] Cell 101
print("Let's extract the information in the TMDB dataset...", end=' ')
data_tmdb = pd.DataFrame([result for results in data_tmdb['results'] for result in results])
print("Done")

print(f"TMDB dataset: {data_tmdb.shape}")

len_tmdb = len(data_tmdb)

# %% [code] Cell 102
data_tmdb.info()

# %% [markdown] Cell 103
# Make a comparison between the two datasets, the `data_movies` originally used and the one downloaded through the TMDB API.

# %% [code] Cell 104
data_tmdb[data_tmdb['title'] == 'The Evil Within']

# %% [code] Cell 105
data_movies[data_movies['Title'] == 'The Evil Within']

# %% [markdown] Cell 106
# As we can see, the request to TMDB by movie title has yield much more movies that what we actually needed. This is to be expected since it's not rare for movies to share the same title, the matter or bringing together the data obtained with TMDB and the one we already have will be solved afterward.
# 
# Now we keep our focus on the attributes, because we can observe that what we obtained from TMDB database doesn't have all the attributes cited in the paper, namely:

# %% [markdown] Cell 107
# The integrated dataset should have the following features for the movies:
# 
# ![image.png](data:image/png;base64,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)

# %% [markdown] Cell 108
# In the data obtained we have information like `backdrop_path`, that in which we are not interested in and can we can simply drop them.
# 
# On the other hand, columns that we would like to have, like actors, runtime, director, ecc., are not present so now we proceed to make other request to the TMDb database to obtain more information on the obtained movies.

# %% [markdown] Cell 109
# ### Get all the movie's attributes from TMDb

# %% [markdown] Cell 110
# To obtain the additional attributes we are interested in we make a different API request to the TMDB database using the TMDB ID in `tmdb_data` obtained by the first API call.

# %% [markdown] Cell 111
# This code block calls two functions: `download_data_by_movie_id` and `download_credits`, each with specific parameters for file paths, the DataFrame `X`, and an API key. The `download_data_by_movie_id` function likely retrieves additional attributes for movies using their IDs, while `download_credits` fetches credit information such as cast and crew. These functions enhance the `X` DataFrame with enriched movie data from an external source.

# %% [code] Cell 112
download_data_by_movie_id(file_path=movie_tmdb_attributes_path, df=X, key=api_key)
download_credits(file_path=movie_credits_path, df=X, key=api_key)

# %% [markdown] Cell 113
# ### Make an association table with the IMDb ids used in the Tweetings DB

# %% [markdown] Cell 114
# In the new attributes we got from TMDB there's also and `imdb_id` field which we can use to extract to create an association table of the id used by TMDB with the id used by IMDB, which is the same that Tweetings DB uses.

# %% [markdown] Cell 115
# This code block imports a JSON file located at `movie_tmdb_attributes_path` into a DataFrame called `data_attributes` using `pd.read_json()`. It prints a message before and after the import to indicate progress and then outputs the shape of the `data_attributes` DataFrame. The variable `len_attributes` is updated with the number of records in the `data_attributes` DataFrame, providing insight into the dataset's size.

# %% [code] Cell 116
print("Let's import the Attributes dataset...", end=' ')
data_attributes = pd.read_json(movie_tmdb_attributes_path)
print("Done")

print(f"Attributes: {data_attributes.shape}")

len_attributes = len(data_attributes)

# %% [code] Cell 117
data_attributes.info()

# %% [markdown] Cell 118
# This code block cleans the `data_attributes` DataFrame by filtering out rows with missing or empty IMDB IDs, then extracts and converts the IMDB IDs to integers by removing the `tt` prefix. The resulting DataFrame, `data_associations`, retains only the `id` and cleaned `imdb_id` columns. It prints the percentage of records kept from the original dataset and updates `len_association` with the number of records in the cleaned DataFrame.

# %% [code] Cell 119
print("Let's clean and extract the IMDb IDs from the Attributes dataset...", end=' ')
data_associations = (
    data_attributes.dropna(subset=['imdb_id'])
    .loc[lambda df: df['imdb_id'] != '']
    .assign(imdb_id=lambda df: df['imdb_id'].str.replace('tt', '', regex=False).astype(int))
    [['id', 'imdb_id']]
)
print("Done")

print(f"We kept {data_associations.shape[0] / len_attributes * 100:.0f}% of the dataset.")
print(f"Associations: {data_associations.shape}")

len_association = len(data_associations)

# %% [code] Cell 120
data_associations.info()

# %% [markdown] Cell 121
# This code block removes duplicate rows from the `data_associations` DataFrame using `drop_duplicates()`, ensuring that only unique rows remain. It then prints the percentage of records retained from the previous dataset and the updated shape of `data_associations`. The `len_association` variable is updated to reflect the number of records after removing duplicates.

# %% [code] Cell 122
print("Let's drop the duplicated rows...", end=' ')
data_associations.drop_duplicates(inplace=True, ignore_index=True)
print("Done")

print(f"We kept {data_associations.shape[0] / len_association * 100:.0f}% of the dataset.")
print(f"Associations: {data_associations.shape}")

len_association = len(data_associations)

# %% [code] Cell 123
data_associations.info()

# %% [markdown] Cell 124
# This code block filters the `data_associations` DataFrame to retain only the movies that are present in the `data_movies` dataset by merging on `MovieID` and `imdb_id`. It then removes the `MovieID` column and renames `id` to `tmdb_id` for clarity. The code prints the percentage of records retained from the previous dataset and updates `len_association` with the new number of records.

# %% [code] Cell 125
print("Let's retain only the movies in the Movies dataset...", end=' ')
data_associations = (data_movies[['MovieID']]
                     .merge(data_associations, left_on='MovieID', right_on='imdb_id')
                     .drop(columns='MovieID')
                     .rename(columns={'id': 'tmdb_id'}))
print("Done")

print(f"We kept {data_associations.shape[0] / len_association * 100:.0f}% of the dataset.")
print(f"Associations: {data_associations.shape}")

len_association = len(data_associations)

# %% [code] Cell 126
data_associations.info()

# %% [markdown] Cell 127
# Check if a file at `tmdb_imdb_association_path` already exists. If the file does not exist, it saves the `data_associations` DataFrame to a JSON file with pretty-printing. If the file already exists, it simply prints a message indicating that the file is already present.

# %% [code] Cell 128
print("Let's save the TMDB/IMDB associations dataset...", end=' ')
if not os.path.exists(tmdb_imdb_association_path):
    data_associations.to_json(tmdb_imdb_association_path, indent=4)
    print('Done')
else:
    print('File already exists')

# %% [markdown] Cell 129
# ### Load the movie's attributes and preprocess the dataset

# %% [markdown] Cell 130
# This code block filters the `data_attributes` DataFrame to retain only the rows where the `id` exists in the `data_associations` DataFrame's `tmdb_id` column. It then resets the index of the filtered DataFrame and prints the percentage of records kept from the original dataset. Finally, it updates `len_attributes` with the number of records remaining after the filter.

# %% [code] Cell 131
print("Let's keep only the movies with a corresponding instance in Tweetings DB...", end=' ')
data_attributes = data_attributes[data_attributes['id'].isin(data_associations['tmdb_id'])].reset_index(drop=True)
print("Done")

print(f"We kept {data_attributes.shape[0] / len_attributes * 100:.0f}% of the dataset.")
print(f"Associations: {data_attributes.shape}")

len_attributes = len(data_attributes)

# %% [code] Cell 132
data_attributes.info()

# %% [markdown] Cell 133
# This code block drops several columns from the `data_attributes` DataFrame that are deemed unnecessary for the task, simplifying the dataset. It then processes the `production_companies` and `production_countries` columns by extracting the `name` field from each dictionary within these columns. Finally, it prints the percentage of records retained from the original dataset and updates `len_attributes` with the new number of records.

# %% [code] Cell 134
print("Let's keep only the important columns for the task...", end=' ')
cols_to_drop = ['adult', 'backdrop_path', 'belongs_to_collection', 'genres', 'homepage', 'imdb_id',
                'origin_country', 'original_title', 'overview', 'poster_path', 'release_date',
                'revenue', 'spoken_languages', 'status', 'tagline', 'video', 'vote_average', 'vote_count']

data_attributes.drop(columns=cols_to_drop, inplace=True)
data_attributes[['production_companies', 'production_countries']] = data_attributes[
    ['production_companies', 'production_countries']].applymap(lambda x: [d['name'] for d in x])
print("Done")

print(f"We kept {data_attributes.shape[0] / len_attributes * 100:.0f}% of the dataset.")
print(f"Attributes: {data_attributes.shape}")

len_attributes = len(data_attributes)

# %% [code] Cell 135
data_attributes.info()

# %% [markdown] Cell 136
# This code block removes duplicate entries in the `data_attributes` DataFrame based on the `id` column, ensuring each movie has a unique identifier. It prints the percentage of records retained from the original dataset after deduplication. Finally, the `len_attributes` variable is updated to reflect the number of records remaining in the cleaned DataFrame.

# %% [code] Cell 137
print("Let's drop entries with duplicated 'id'...", end=' ')
data_attributes.drop_duplicates(subset='id', inplace=True)
print("Done")

print(f"We kept {data_attributes.shape[0] / len_attributes * 100:.0f}% of the dataset.")
print(f"Attributes: {data_attributes.shape}")

len_attributes = len(data_attributes)

# %% [markdown] Cell 138
# This code block imports a JSON file located at `movie_credits_path` into a DataFrame called `data_credits` using `pd.read_json()`. It prints a message indicating the completion of the import and the shape of the `data_credits` DataFrame. Additionally, it updates `len_credits` with the number of records in the `data_credits` DataFrame.

# %% [code] Cell 139
print("Let's import the Credits dataset...", end=' ')
data_credits = pd.read_json(movie_credits_path)
print("Done")

print(f"Credits: {data_credits.shape}")

len_credits = len(data_credits)

# %% [code] Cell 140
data_credits.info()

# %% [markdown] Cell 141
# This code block first filters the `data_credits` DataFrame to retain only the rows where the `id` is present in the `data_associations` DataFrame's `tmdb_id`, and then resets the DataFrame's index. It prints the percentage of records retained and updates `len_credits`. Afterward, it removes duplicate rows based on the `id` column and prints the updated percentage and shape of the `data_credits` DataFrame.

# %% [code] Cell 142
print("Let's keep only the credits of movies in the Association dataset...", end=' ')
data_credits = data_credits[data_credits.id.isin(data_associations['tmdb_id'])].reset_index(drop=True)
print("Done")

print(f"We kept {data_credits.shape[0] / len_credits * 100:.0f}% of the dataset.")
print(f"Credits: {data_credits.shape}")

len_credits = len(data_credits)

print("Let's drop rows with duplicated 'id'...", end=' ')
data_credits.drop_duplicates(subset=['id'], inplace=True)
print("Done")

print(f"We kept {data_credits.shape[0] / len_credits * 100:.0f}% of the dataset.")
print(f"Credits: {data_credits.shape}")

len_credits = len(data_credits)

# %% [markdown] Cell 143
# This code block extracts information from the `cast` feature of the `data_credits` DataFrame. It creates new columns: `actors` containing actor names, `actors_popularity` containing their popularity scores, `directors` containing names of directors, and `writers` containing names of writers. It prints the shape of the `data_credits` DataFrame after these transformations.

# %% [code] Cell 144
print("Let's extract the information from the 'cast' feature...", end=' ')
data_credits['actors'] = data_credits['cast'].apply(lambda x: [d['name'] for d in x])
data_credits['actors_popularity'] = data_credits['cast'].apply(lambda x: [d['popularity'] for d in x])
data_credits['directors'] = data_credits['cast'].apply(
    lambda x: [d['name'] for d in x if d['known_for_department'] == 'Directing'])
data_credits['writers'] = data_credits['cast'].apply(
    lambda x: [d['name'] for d in x if d['known_for_department'] == 'Writing'])
print("Done")

print(f"Credits: {data_credits.shape}")

# %% [code] Cell 145
data_credits.info()

# %% [markdown] Cell 146
# ### Merge the TMDB data, to gather all movies information

# %% [markdown] Cell 147
# This code block merges the `data_attributes` and `data_credits` DataFrames on the `id` column to create a unified TMDB dataset. The `how='inner'` parameter ensures that only rows with matching `id` values in both DataFrames are included. It then prints the shape of the resulting DataFrame and updates the `len_tmdb` variable with the number of rows.

# %% [code] Cell 148
print("Let's merge the Attributes and Credits dataset into a TMDB dataset...", end=' ')
X = data_attributes.merge(data_credits, on='id', how='inner')
print("Done")

print(f"TMDb: {X.shape}")

len_tmdb = len(X)

# %% [code] Cell 149
X.info()

# %% [markdown] Cell 150
# The code block checks if a file at `tmdb_movies_path` already exists; if it does not, it saves the `X` DataFrame to a JSON file with indents for readability. If the file does exist, it skips the save operation and prints a message indicating that the file is already present. This helps avoid overwriting existing data while ensuring the dataset is saved correctly.

# %% [code] Cell 151
print("Let's save the TMDB dataset...", end=' ')
if not os.path.exists(tmdb_movies_path):
    X.to_json(tmdb_movies_path, indent=4)
    print('Done')
else:
    print('File already exists')

# %% [markdown] Cell 152
# The code block merges the `X` DataFrame (containing TMDB data) with the `data_associations` DataFrame on the `id` and `tmdb_id` columns, aligning TMDb data with IMDB identifiers. After merging, it drops redundant columns and renames `imdb_id` to `MovieID` for consistency. This ensures that the resulting dataset contains aligned and correctly labeled movie identifiers from IMDB.

# %% [code] Cell 153
print("Let's merge the TMDB dataset with the Association dataset...", end=' ')
X = pd.merge(X, data_associations, left_on='id', right_on='tmdb_id')
X.drop(['id', 'tmdb_id'], axis=1, inplace=True)
X.rename(columns={'imdb_id': 'MovieID'}, inplace=True)
print("Done")

print(f"IMDb: {X.shape}")

len_X = len(X)

# %% [code] Cell 154
X.info()

# %% [markdown] Cell 155
# The code block checks if the file at `imdb_movies_path` already exists; if it does not, it saves the `X` DataFrame (now containing the merged TMDB and IMDB data) as a JSON file at that path. If the file exists, it avoids overwriting by printing a message. This ensures that the processed dataset is saved only if it hasn't been previously saved.

# %% [code] Cell 156
print("Let's save the IMDB dataset...", end=' ')
if not os.path.exists(imdb_movies_path):
    X.to_json(imdb_movies_path, indent=4)
    print('Done')
else:
    print('File already exists')
