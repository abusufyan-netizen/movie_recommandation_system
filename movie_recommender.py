"""
Movie Recommendation System using Python
-----------------------------------------
This system recommends movies similar to the one entered by the user.
It uses content-based filtering with CountVectorizer and Cosine Similarity.

Author: Abuzar
Language: Python 3.x
Libraries: pandas, numpy, scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# -----------------------------
# STEP 1: Load and merge dataset
# -----------------------------
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')

# -----------------------------
# STEP 2: Select and clean data
# -----------------------------
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

def convert(obj):
    """Convert stringified list of dictionaries to a list of names"""
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def fetch_director(obj):
    """Extract director name from crew data"""
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def convert_cast(obj):
    """Extract top 3 cast members"""
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L

# Apply transformations
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# -----------------------------
# STEP 3: Create a new combined feature 'tags'
# -----------------------------
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# -----------------------------
# STEP 4: Convert text data to vectors
# -----------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Compute cosine similarity between vectors
similarity = cosine_similarity(vectors)

# -----------------------------
# STEP 5: Recommendation function
# -----------------------------
def recommend(movie):
    """
    Recommend 5 similar movies based on the given movie title.
    """
    movie = movie.lower().strip()
    if movie not in new_df['title'].str.lower().values:
        print(f"\n❌ Sorry, '{movie}' not found in the database. Try another title.\n")
        return
    
    index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    print("\n🎬 Top 5 Recommended Movies:\n")
    for i, (idx, score) in enumerate(movie_list, start=1):
        print(f"{i}. {new_df.iloc[idx].title}  (Similarity Score: {round(score, 2)})")
    print("\n----------------------------------\n")

# -----------------------------
# STEP 6: Main Execution
# -----------------------------
if __name__ == "__main__":
    print("====================================")
    print("   🎥  MOVIE RECOMMENDER SYSTEM  🎥  ")
    print("====================================")
    print("Example movies: Inception, Avatar, Batman Begins, Titanic, Gladiator\n")

    while True:
        movie_name = input("Enter a movie name (or type 'exit' to quit): ")
        if movie_name.lower() == 'exit':
            print("\n👋 Thank you for using the Movie Recommender System!")
            break
        recommend(movie_name)
