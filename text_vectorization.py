import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Load cleaned data
data = pd.read_csv('data/cleaned_spam.csv')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(data['cleaned_message']).toarray()
y = data['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save the vectorizer
joblib.dump(tfidf, 'app/vectorizer.pkl')
print("✅ Text Vectorization Completed")
