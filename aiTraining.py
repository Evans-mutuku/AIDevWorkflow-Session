# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Download necessary NLTK data
nltk.download('stopwords')


def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing punctuation and stopwords
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Simple tokenization (split by whitespace)
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Sample data
data = {
    'review': [
        'I love this product!',
        'This is the worst product I have ever bought.',
        'Great quality and fast delivery.',
        'Not worth the money.',
        'Highly recommend this product.'
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
}

df = pd.DataFrame(data)
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Convert text data to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Create DataFrame and process


print("\nProcessed Reviews:")
print(df[['review', 'cleaned_review']])

# Example usage of calculate_sum # Output: Sum of numbers: 15