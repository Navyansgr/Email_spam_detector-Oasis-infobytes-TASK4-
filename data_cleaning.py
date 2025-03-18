import pandas as pd
import re

# Load dataset
data = pd.read_csv('data/spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]  # Keeping only relevant columns
data.columns = ['label', 'message']

# Encode labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)       # Remove numbers
    return text

data['cleaned_message'] = data['message'].apply(clean_text)

# Save cleaned data
data.to_csv('data/cleaned_spam.csv', index=False)
print("✅ Data Cleaning Completed")
