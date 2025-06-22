from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Muat model dan vectorizer
model = joblib.load('model_rf_hs.joblib')
vectorizer = joblib.load('vectorizer_hs.joblib')

# Preprocessing sederhana
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # hapus tanda baca
    text = re.sub(r'\d+', '', text)      # hapus angka
    text = text.lower().strip()          # huruf kecil & trim spasi
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    cleaned_text = clean_text(input_text)
    transformed = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed)

    # Ubah ke label (opsional, bisa disesuaikan dengan label asli)
    sentiment = 'Hate Speech' if prediction[0] == 1 else 'Not Hate Speech'

    return render_template('index.html', prediction=sentiment, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
