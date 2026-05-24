from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = 'model.pkl'

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Machine learning model successfully loaded.")
else:
    model = None
    print("⚠️ model.pkl not found! Please run train.py first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Machine learning model file is missing.'}), 500

    try:
        data = request.get_json()
        age = float(data['age'])
        bmi = float(data['bmi'])
        goal = int(data['goal'])

        features = np.array([[age, bmi, goal]])
        prediction = model.predict(features)[0]

        return jsonify({'recommendation': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
