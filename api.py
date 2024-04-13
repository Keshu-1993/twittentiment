from flask import Flask, request, jsonify
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from data_prep import preprocess_text

# Import necessary libraries
from data_prep import train_data

app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to the sentiment analysis API!'

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer with training data
X_train = tfidf_vectorizer.fit_transform(train_data['Tweet_Content'])

# Load trained model
model = torch.load('/Users/keshav/Documents/For_Interviews/ClLo_tech/saved_model/trained_model.pth')

label_mapping = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'Text field is missing in the request'}), 400
    text = data['text']
    if not text.strip():
        return jsonify({'error': 'Text is empty or contains only whitespace'}), 400

    preprocessed_text = preprocess_text(text)
    features = tfidf_vectorizer.transform([preprocessed_text])
    
    # Load the model in evaluation mode
    model.eval()

    with torch.no_grad():
        output = model(torch.Tensor(features.toarray()))
        _, predicted = torch.max(output, 1)
    sentiment = predicted.item()
    sentiment_label = list(label_mapping.keys())[list(label_mapping.values()).index(sentiment)]
    return jsonify({'The sentiment of your text is': sentiment_label})

if __name__ == '__main__':
    app.run(debug=True)
