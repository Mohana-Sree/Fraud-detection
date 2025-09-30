from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# --- Load The Model and Scaler ---
model_path = os.path.join('model', 'fraud_detection_model.pkl')
print(f"Loading model and scaler from {model_path}...")
try:
    with open(model_path, 'rb') as file:
        # Load the dictionary containing the model and scaler
        data = pickle.load(file)
        model = data['model']
        scaler = data['scaler']
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please run model_training.py first.")
    model = None
    scaler = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model is not loaded. Please check the server logs.'})

    try:
        # Get the form data
        features = [
            float(request.form['v1']),
            float(request.form['v2']),
            float(request.form['v3']),
            float(request.form['v4']),
            float(request.form['v5']),
            float(request.form['amount'])
        ]
        
        # Convert to a numpy array and reshape for a single prediction
        final_features = np.array(features).reshape(1, -1)
        
        # **IMPORTANT: Scale the features using the loaded scaler**
        scaled_features = scaler.transform(final_features)
        
        # Make prediction on the scaled features
        prediction_proba = model.predict_proba(scaled_features)
        
        # Get the confidence score for the predicted class
        is_fraud = model.predict(scaled_features)[0] == 1
        confidence = prediction_proba[0][1] if is_fraud else prediction_proba[0][0]

        # Prepare result message
        result_text = "This transaction is likely FRAUDULENT" if is_fraud else "This transaction is likely LEGITIMATE"
        full_result = f"{result_text} with a {confidence*100:.2f}% confidence."

        return jsonify({
            'result': full_result,
            # **THE FIX IS HERE: Convert the boolean to a standard Python bool**
            'is_fraud': bool(is_fraud)
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Invalid input data. Please ensure all fields are numbers.'})

if __name__ == '__main__':
    app.run(debug=True)

