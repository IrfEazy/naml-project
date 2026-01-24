<a name="readme-top"></a>

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org) [![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) [![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org) [![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)](https://scipy.org) [![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org) [![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![Seaborn](https://img.shields.io/badge/seaborn-v0.13.2-blue?style=for-the-badge)](https://seaborn.pydata.org) [![Surprise](https://img.shields.io/badge/scikit--surprise-v1.1.4-9cf?style=for-the-badge)](https://surpriselib.com)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/IrfEazy/naml-project">
    <img src="icon.png" alt="Icon" width="256">
  </a>
  <h3 align="center">NAML Project</h3>

  <p align="center">
    Numerical Analysis For Machine Learning - Movie Recommendation System using Sentiment Analysis from Microblogging Data
    <br />
    Academic Year 2023-2024
    <br />
    <br />
  </p>
  <br />
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

NAML is a comprehensive movie recommendation system that leverages sentiment analysis data from the MovieTweetings dataset, enriched with metadata from The Movie Database (TMDB) API. The project implements and compares multiple recommendation approaches to provide personalized movie suggestions based on user preferences, movie content, and public sentiment.

### Project Overview

This project develops several movie recommender systems following the research outlined in "Movie Recommendation System Using Sentiment Analysis from Microblogging Data." Instead of performing sentiment analysis from scratch, we utilized pre-existing sentiment scores from the MovieTweetings dataset and augmented them with rich metadata from the TMDB API to create a robust hybrid recommendation engine.

### Key Components

**1. Data Preprocessing and Analysis**
- Initial data processing of the MovieTweetings dataset containing user ratings and movie metadata
- Data augmentation through TMDB API integration to enhance movie features
- Comprehensive exploratory data analysis including genre distribution, rating patterns, and temporal trends
- Final dataset filtering focused on movies released between 2014-2017

**2. Collaborative Filtering Recommendation System**
- **Most Popular Recommender**: Suggests movies with the highest average ratings across users, using a shrinkage factor to penalize low-rating-count movies
- **Funk SVD Matrix Factorization**: Implements matrix factorization using Stochastic Gradient Descent to uncover latent factors in user-movie interactions

**3. Content-Based Recommendation System**
- **Simple CB Recommender**: Recommends movies based on feature similarity using TF-IDF vectorization and cosine similarity
- **Enhanced CB Recommender**: Augments feature vectors with movie overviews and additional metadata from TMDB to improve recommendation relevance

**4. Hybrid Recommendation System**
- **Mixed Hybrid Approach**: Combines CF and CB recommendations using intersection and union operations to balance user preferences with content similarity
- **Meta-Level Hybrid Approach**: Uses CF recommendations as input for CB refinement, creating a cascading recommendation pipeline

**5. Model Evaluation**
- Quantitative evaluation using precision metrics on train-test splits
- Qualitative analysis of recommendation relevance and user-system alignment
- Comparative analysis of all implemented systems

## Dataset

The project utilizes two primary data sources:

**MovieTweetings Dataset**
- 37,342 movies with metadata (title, release year, genres)
- User ratings on a scale of 1-10 with timestamps
- Pre-computed sentiment scores from Twitter/X microblogging data

**The Movie Database (TMDB) API**
- Movie details: overviews, release dates, genres
- Credits information: cast, directors, producers
- Production metadata: companies, countries, languages

**Final Dataset Characteristics (2014-2017)**
- 28 distinct movie genres after filtering
- Users with minimum 20 ratings to ensure engagement
- Temporal distribution showing seasonal patterns in reviews

## Key Findings

### Recommendation System Performance
The evaluation revealed that individual recommender systems (CF and CB) outperform hybrid approaches in terms of precision. This is attributed to data sparsity challenges and the difficulty in effectively combining heterogeneous recommendation signals.

### Content-Based Filtering Insights
Despite augmenting the feature vector with movie overviews (increasing from 53,904 to 66,974 features), the CB system produced largely similar recommendations. This suggests that genre and cast information dominate similarity computations, while textual descriptions have limited additional impact.

### Data Characteristics
- **Rating Skew**: Majority of ratings cluster around 7-8 out of 10, introducing bias toward highly-rated movies
- **User Sparsity**: Most users provided approximately 25 ratings, creating a sparse user-item interaction matrix
- **Temporal Patterns**: Noticeable peaks in review activity coinciding with major film releases and year-end reflections

## Methodology

### Data Processing Pipeline
1. Duplicate removal from movies dataset
2. Genre conversion to structured list format
3. Filtering movies by genre frequency (minimum 20 occurrences)
4. Title and release year extraction using regex
5. Release year filtering (2014-2017)
6. User engagement filtering (minimum 20 ratings)
7. TMDB API enrichment for additional metadata

### Feature Engineering
- TF-IDF vectorization for text-based features
- Cosine similarity matrix generation for content similarity
- User-item interaction matrix (URM) creation for collaborative filtering

### Model Training
- Funk SVD implementation using Stochastic Gradient Descent
- Ridge regularization to prevent overfitting
- Train-test split for model evaluation

## Results Summary

### Precision@5 Comparison

| Recommender System | Precision@5 |
| ------------------ | ----------- |
| Most Popular (CF)  | ~0.0469     |
| Top Recommendation | ~0.0401     |
| Content-Based      | ~0.0318     |
| Mixed Hybrid       | ~0.0172     |
| Meta-Level Hybrid  | ~0.0078     |

The modest precision values reflect the challenges of working with sparse data and the inherent difficulty of predicting user preferences from limited interaction history.

## References

1. GitHub repository: https://github.com/IrfEazy/naml-project
2. MovieTweetings Dataset: https://github.com/sidooms/MovieTweetings
3. The Movie Database (TMDB) API: https://www.themoviedb.org/settings/api
