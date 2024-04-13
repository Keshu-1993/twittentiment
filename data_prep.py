import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# def preprocess_text(text):
#     # Convert text to lowercase
#     text = text.lower()
#     # Remove special characters and digits
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     # Tokenize text
#     tokens = word_tokenize(text)
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     filtered_tokens = [word for word in tokens if word not in stop_words]
#     # Lemmatize words
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
#     # Join tokens back into text
#     preprocessed_text = ' '.join(lemmatized_tokens)
#     return preprocessed_text

def preprocess_text(text):
    # Check if text is NaN
    if pd.isna(text):
        return ''
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Load data
def load_data(file_path):
    # Load data without header
    data = pd.read_csv(file_path, header=None, names=['Tweet_id', 'Entity', 'Sentiment', 'Tweet_Content'])
    # Filter out rows with "Irrelevant" sentiment
    filtered_data = data[data['Sentiment'] != 'Irrelevant']
    #print(set(filtered_data['Sentiment']))
    return filtered_data

def preprocess_data(data):
    # Apply preprocessing to text data
    data['Tweet_Content'] = data['Tweet_Content'].apply(preprocess_text)
    # Map sentiment labels to numeric values
    label_mapping = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
    data['Sentiment'] = data['Sentiment'].map(label_mapping)
    return data

train_data = load_data('/Users/keshav/Documents/For_Interviews/ClLo_tech/Dataset/twitter_training.csv')
validation_data = load_data('/Users/keshav/Documents/For_Interviews/ClLo_tech/Dataset/twitter_validation.csv')

# Preprocess data
train_data = preprocess_data(train_data)
validation_data = preprocess_data(validation_data)

# Apply preprocessing to text data
#train_data['text'] = train_data['text'].apply(preprocess_text)
#validation_data['text'] = validation_data['text'].apply(preprocess_text)

