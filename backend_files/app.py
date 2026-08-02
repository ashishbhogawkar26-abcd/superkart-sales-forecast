from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import io

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('superkart_model.joblib')

@app.route('/v1/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert to DataFrame (model expects same features as training)
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)

        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/v1/predictbatch', methods=['POST'])
def predict_batch():
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        # Read CSV file
        df = pd.read_csv(io.BytesIO(file.read()))

        # Make predictions
        predictions = model.predict(df)

        # Return predictions as a dictionary mapping index to result
        result = {int(i): float(p) for i, p in enumerate(predictions)}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run on port 7860 as specified in Dockerfile
    app.run(host='0.0.0.0', port=7860)
