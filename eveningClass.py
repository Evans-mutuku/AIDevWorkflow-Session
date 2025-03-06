# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice

# Sample data: Customer reviews
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

# Create a DataFrame
df = pd.DataFrame(data)

nltk.download('punkt')
nltk.download('stopwords')

# Sample text preprocessing function
def preprocess_text(text):
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Simple tokenization (split by whitespace)
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the DataFrame
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


# Load the workspace
ws = Workspace.from_config()

# Register the model
model = Model.register(workspace=ws, model_path='model.pkl', model_name='sentiment_analysis_model')

# Deploy the model as a web service
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
service = Model.deploy(workspace=ws, name='sentiment-analysis-service', models=[model], inference_config=None, deployment_config=aci_config)
service.wait_for_deployment(show_output=True)

# Test the deployed service
print(service.scoring_uri)

print(df[['review', 'cleaned_review']])
print(df)