from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load data
import pandas as pd
data = pd.read_csv('data/cleaned_spam.csv')
X = joblib.load('app/vectorizer.pkl').transform(data['cleaned_message']).toarray()
y = data['label']

# Train Model
model = LogisticRegression()
model.fit(X, y)

# Save Model
joblib.dump(model, 'app/model.pkl')
print("✅ Model Training Completed")
