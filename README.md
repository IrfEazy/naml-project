# Movie Recommendation System Using Sentiment Analysis from Microblogging Data

## Project Overview

This project develops multiple movie recommendation systems leveraging the MovieTweetings dataset enriched with metadata from the TMDB API. The systems implement collaborative filtering, content-based filtering, and hybrid approaches to provide personalized movie recommendations based on user ratings and movie attributes.

## Key Features

- **Collaborative Filtering Systems**: Most Popular recommender and Funk-SVD matrix factorization approach
- **Content-Based Filtering**: Feature vectors combining genres, cast, directors, writers, production companies, languages, and movie overviews using TF-IDF transformation
- **Hybrid Recommenders**: Mixed Hybrid (union/intersection of CF and CB) and Meta-Level Hybrid (cascading CF output through CB) systems
- **Data Augmentation**: Integration of TMDB API metadata to enrich the MovieTweetings dataset with director, cast, runtime, and plot information
- **Comprehensive Evaluation**: Qualitative analysis and precision@5 metrics with train-test split methodology

## Technical Stack

- **Language**: Python
- **Primary Dataset**: MovieTweetings (37,342 movies initially)
- **External API**: TMDB (The Movie Database)
- **ML Library**: Surprise library for Funk-SVD implementation
- **Feature Processing**: TF-IDF vectorization, cosine similarity matrices
- **Data Filtering**: Movies filtered to 2014-2017 release years; genres with minimum 20 instances retained

## Main Results

### Precision@5 Performance
- Mixed Hybrid Recommender: 0.0172
- Meta-Level Hybrid Recommender: 0.0078
- Individual CF and CB systems outperformed hybrid approaches

### Dataset Analysis
- 28 distinct genres identified; filtered to focus on genres with 20+ movie instances
- Ratings distribution skewed toward higher scores (modes at 7-8)
- Majority of users provided approximately 25 ratings each
- Temporal analysis revealed peaks in review activity aligned with major film releases

### Key Observations
- Hybrid systems underperformed individual recommenders due to data sparsity and integration challenges
- Most Popular and Funk-SVD approaches achieved comparable performance, indicating data sparsity limitations prevent advanced methods from differentiating performance
- Content-Based system recommendations reasonably aligned with user preferences, though sometimes recommended less intuitive films due to attribute-based similarity

## Limitations

- **Evaluation Constraints**: TMDB ground-truth recommendations spanned 1980s-present, creating fundamental mismatch with model's 2014-2017 training timeframe with minimal overlap (ground-truth unusable for model evaluation)
- **Data Sparsity**: Limited and noisy user interaction data, particularly for less-reviewed movies, significantly impacted model performance
- **Incomplete Attributes**: Movies with single or very few ratings posed challenges for recommendation accuracy
- **Precision Metric Limitation**: Evaluation methodology penalizes recommending relevant movies not present in test set, potentially underestimating true performance
- **Hybrid System Integration**: Overlapping limitations of CF and CB methods amplified through combination rather than resolved

## Project Structure

```
├── src/
│   ├── analyzer.py           # Data analysis utilities
│   ├── cleaner.py            # Data preprocessing
│   ├── config.py             # Configuration management
│   ├── evaluation.py         # Model evaluation metrics
│   ├── loader.py             # Data loading functions
│   ├── tmdb.py               # TMDB API integration
│   ├── visualization.py      # Visualization utilities
│   └── recommenders/
│       ├── base.py           # Base recommender class
│       ├── baseline.py       # Most Popular recommender
│       ├── collaborative.py  # Collaborative filtering (Funk-SVD)
│       ├── content_based.py  # Content-based recommender
│       └── hybrid.py         # Hybrid recommender systems
├── data/                      # Dataset files
├── docs/                      # Documentation and thesis
├── data_analysis.py          # Analysis notebook/script
└── movie_rec_sys.py          # Main recommendation system
```

## Source Material

This README is derived from the LaTeX report located in the `docs/` folder and the project implementation.