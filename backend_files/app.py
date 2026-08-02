from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import io
# Import necessary scikit-learn components for preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model, preprocessor, and the expected input feature order
model = joblib.load('superkart_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')
input_features_order = joblib.load('input_features_order.joblib')

# Extract feature names after one-hot encoding for creating the final DataFrame
# This assumes the preprocessor has already been fitted on the training data
# We need to manually reconstruct the all_feature_names from the preprocessor
# Numerical columns will be in the same order as in input_features_order
# Categorical columns will be one-hot encoded and appended

# Get numerical columns that were scaled
numerical_cols_processed = [col for col in input_features_order if col in ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age_Years']]

# Get one-hot encoded feature names
categorical_cols_for_ohe = [col for col in input_features_order if col in ['Product_Sugar_Content', 'Store_Id', 'Store_Size', 'Store_Location_City_Type', 'Store_Type', 'Product_Id_char', 'Product_Type_Category']]
ohe_feature_names = preprocessor.named_transformers_['onehotencoder'].get_feature_names_out(categorical_cols_for_ohe)

all_feature_names = numerical_cols_processed + list(ohe_feature_names)

@app.route('/v1/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert to DataFrame, ensuring all expected columns are present and in order
        df_raw = pd.DataFrame([data], columns=input_features_order)

        # Preprocess the raw input data using the loaded preprocessor
        processed_data_array = preprocessor.transform(df_raw)

        # Convert the processed numpy array back to a DataFrame with correct column names
        df_processed = pd.DataFrame(processed_data_array, columns=all_feature_names)

        # Make prediction
        prediction = model.predict(df_processed)

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
        df_raw = pd.read_csv(io.BytesIO(file.read()))

        # Ensure batch data has all expected columns and in correct order for preprocessing
        df_raw = df_raw[input_features_order] # Reorder to match input_features_order

        # Preprocess the raw batch data
        processed_data_array = preprocessor.transform(df_raw)

        # Convert the processed numpy array back to a DataFrame with correct column names
        df_processed = pd.DataFrame(processed_data_array, columns=all_feature_names)

        # Make predictions
        predictions = model.predict(df_processed)

        # Return predictions as a dictionary mapping index to result
        result = {int(i): float(p) for i, p in enumerate(predictions)}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run on port 7860 as specified in Dockerfile
    app.run(host='0.0.0.0', port=7860)
