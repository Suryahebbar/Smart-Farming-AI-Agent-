from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model
with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Home route renders input form
@app.route('/')
def home():
    return render_template('index.html')

# Handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form.get(k)) for k in [
        'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
    ]]
    prediction = model.predict([features])[0]
    return render_template('index.html', prediction_text=f'Recommended Crop: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
