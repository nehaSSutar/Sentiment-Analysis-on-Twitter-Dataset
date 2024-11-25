from flask import Flask, render_template, request
import pickle
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import numpy as np
app = Flask(__name__)

# Load sentiment analysis model and its vectorizer
with open('final_model.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    sentiment_vectorizer = pickle.load(f)

# Load keyword analysis model and its vectorizer
with open('keyword_sentiment_model.pkl', 'rb') as f:
    keyword_model = pickle.load(f)

with open('keyword_vectorizer.pkl', 'rb') as f:
    keyword_vectorizer = pickle.load(f)

# Home route (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Route for sentiment analysis page (sentiment_analysis.html)
@app.route('/sentiment_analysis')
def sentiment_analysis():
    return render_template('sentiment_analysis.html')

# Route for keyword analysis page (keyword_analysis.html)
@app.route('/keyword_analysis')
def keyword_analysis():
    return render_template('keyword_analysis.html')

# Sentiment analysis form processing
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    tweet = request.form['tweet']
    # Preprocess the tweet using the sentiment vectorizer
    tweet_vectorized = sentiment_vectorizer.transform([tweet])
    # Perform prediction using the sentiment model
    sentiment = sentiment_model.predict(tweet_vectorized)[0]
    return render_template('sentiment_analysis.html', sentiment=sentiment, tweet=tweet)


#Load the dataset
train_df = pd.read_csv(r'D:\MajorProject\train_data.csv', header=0)
df = pd.DataFrame(train_df)
# Replace NaN values with an empty string
df['Text'] = df['Text'].fillna('')


@app.route('/analyze_keyword', methods=['POST'])
def analyze_keyword():
    keyword = request.form['keyword']  # Get the keyword from the form input

    # Ensure the 'Text' column contains the keyword (case-insensitive)
    filtered_reviews = train_df[train_df['Text'].notna() & train_df['Text'].str.contains(keyword, case=False)]

    if filtered_reviews.empty:
        return f"No reviews found for keyword '{keyword}'."

    # Predicting sentiment for filtered reviews
    filtered_vector = keyword_vectorizer.transform(filtered_reviews['Text'])
    filtered_sentiment = keyword_model.predict(filtered_vector)

    # Map predictions to sentiment labels
    sentiment_mapping = {2: 'Positive', 1: 'Neutral', 0: 'Negative', -1: 'Irrelevant'}
    filtered_reviews['predicted_sentiment'] = filtered_sentiment
    filtered_reviews['predicted_sentiment'] = filtered_reviews['predicted_sentiment'].map(sentiment_mapping)

    # Plot the sentiment distribution
    sentiment_counts = filtered_reviews['predicted_sentiment'].value_counts()
    
    # Generate the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f'Sentiment Analysis for Keyword: {keyword}')
    plt.axis('equal')

    # Convert plot to a base64 image to embed in HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    pie_chart_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('keyword_analysis.html', keyword=keyword, pie_chart_url=pie_chart_url)




if __name__ == "__main__":
    app.run(debug=True)
