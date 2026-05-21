from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies = pd.read_csv('movies_metadata.csv', low_memory=False)

# Fill missing values
movies['title'] = movies['title'].fillna('')
movies['overview'] = movies['overview'].fillna('')

# Remove duplicate titles
movies = movies.drop_duplicates(subset='title')

# Convert text into vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['overview']).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie_name):

    if movie_name not in movies['title'].values:
        return ["Movie not found"]

    movie_index = movies[movies['title'] == movie_name].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend_api():

    movie_name = request.form['movie']

    recommendations = recommend(movie_name)

    return render_template(
        'index.html',
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)
