from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd

# Load Data
data = pd.read_csv('data/cleaned_spam.csv')
X = joblib.load('app/vectorizer.pkl').transform(data['cleaned_message']).toarray()
y = data['label']

# Load Model
model = joblib.load('app/model.pkl')

# Predictions
y_pred = model.predict(X)

# Evaluation
print("🔎 Classification Report:\n", classification_report(y, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y, y_pred))
