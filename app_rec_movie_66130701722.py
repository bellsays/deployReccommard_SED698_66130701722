

import streamlit as st
import pickle
import pandas as pd
from surprise import SVD, Dataset, Reader

# Load data back from the file
with open('recommendation_movie_svd_66130701722.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Streamlit app title
st.title("Movie Recommendation System")

# User input for user_id
user_id = st.number_input("Enter User ID:", min_value=1, value=1)

if st.button("Get Recommendations"):
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
    
    # Predict ratings for unrated movies using the SVD model
    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    
    # Sort predictions by estimated rating in descending order
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
    
    # Get top 10 movie recommendations
    top_recommendations = sorted_predictions[:10]
    
    # Display top recommendations
    st.subheader(f"Top 10 Movie Recommendations for User {user_id}:")
    for recommendation in top_recommendations:
        movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
        st.write(f"{movie_title} (Estimated Rating: {recommendation.est:.2f})")

# Add additional options for user interaction, such as filtering or sorting
