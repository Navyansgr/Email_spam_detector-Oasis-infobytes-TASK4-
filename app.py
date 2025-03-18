from flask import Flask, request, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('app/model.pkl')
vectorizer = joblib.load('app/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_content = request.form['email_content']
    processed_content = vectorizer.transform([email_content])
    prediction = model.predict(processed_content)

    result = "🚨 Spam" if prediction[0] == 1 else "✅ Not Spam"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
