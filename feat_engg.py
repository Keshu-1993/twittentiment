from sklearn.feature_extraction.text import TfidfVectorizer
from data_prep import train_data, validation_data
# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform on training data
X_train = tfidf_vectorizer.fit_transform(train_data['Tweet_Content'])

# Transform validation data
X_validation = tfidf_vectorizer.transform(validation_data['Tweet_Content'])

# Get labels
y_train = train_data['Sentiment']
y_validation = validation_data['Sentiment']

# # Get the count of samples (X) and labels (y) in the training data
# X_train_count = X_train.shape[0]  # Number of samples (rows) in train_data
# y_train_count = y_train.shape[0]  # Number of unique labels in train_data

# # Get the count of samples (X) and labels (y) in the validation data
# X_validation_count = X_validation.shape[0]  # Number of samples (rows) in validation_data
# y_validation_count = y_validation.shape[0]  # Number of unique labels in validation_data

# print("Training data - X count:", X_train_count)
# print("Training data - y count:", y_train_count)
# print("Validation data - X count:", X_validation_count)
# print("Validation data - y count:", y_validation_count)