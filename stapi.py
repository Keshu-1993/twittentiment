import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from data_prep import preprocess_text, train_data

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

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    features = tfidf_vectorizer.transform([preprocessed_text])

    # Load the model in evaluation mode
    model.eval()

    with torch.no_grad():
        output = model(torch.Tensor(features.toarray()))
        _, predicted = torch.max(output, 1)
    sentiment = predicted.item()
    sentiment_label = list(label_mapping.keys())[list(label_mapping.values()).index(sentiment)]
    return sentiment_label

def main():
    st.title("Sentiment Analysis App")

    text = st.text_area("Enter your text here:")
    if st.button("Analyze"):
        if text.strip():
            sentiment = predict_sentiment(text)
            st.write("The sentiment of your text is:", sentiment)
        else:
            st.warning("Text is empty or contains only whitespace. Please enter some text.")

if __name__ == "__main__":
    main()
