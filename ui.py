import streamlit as st
import pandas as pd
import joblib
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("movies.csv")

df.fillna("", inplace=True)
selected_features = ["genres", "keywords", "popularity", "tagline", "title", "cast", "director"]
df["combined_features"] = df[selected_features].apply(lambda x: " ".join(x.astype(str)), axis=1)

# Convert text data to feature vectors
vectorizer = joblib.load("vectorizer.pkl")
feature_vectors = vectorizer.transform(df["combined_features"])

# Compute similarity scores
similarity = cosine_similarity(feature_vectors)

def recommend_movies(movie_name):
    list_of_titles = df["title"].tolist()
    close_match = difflib.get_close_matches(movie_name, list_of_titles, n=1)
    if not close_match:
        return ["No matching movie found."]
    most_close_match = close_match[0]
    index_of_movie = df[df["title"] == most_close_match].index[0]
    similarity_scores = list(enumerate(similarity[index_of_movie]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]
    recommended_movies = [df.iloc[i[0]]["title"] for i in sorted_movies]
    return recommended_movies

# Streamlit UI
st.title("Movie Recommendation System")
movie_name = st.text_input("Enter your favrouite movie name:")
if st.button("Get Recommendations"):
    recommendations = recommend_movies(movie_name)
    st.write("Suggested Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")
