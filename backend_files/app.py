# SuperKart sales-forecasting API.
import io
import logging
import traceback

import joblib
import pandas as pd
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

model = joblib.load('superkart_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')
input_features_order = joblib.load('input_features_order.joblib')
engineered_feature_names = list(preprocessor.get_feature_names_out())

# Allowed values for each one-hot encoded column, read from the FITTED encoder
# so the API and the model can never disagree about what counts as valid.
_ohe = preprocessor.named_transformers_['onehotencoder']
_ohe_cols = [cols for name, _, cols in preprocessor.transformers_ if name == 'onehotencoder'][0]
allowed_categories = {c: list(v) for c, v in zip(_ohe_cols, _ohe.categories_)}
numeric_features = [c for c in input_features_order if c not in allowed_categories]


class BadRequest(Exception):
    pass


@app.errorhandler(BadRequest)
def _bad_request(err):
    return jsonify({'error': str(err)}), 400


@app.errorhandler(Exception)
def _server_error(err):
    app.logger.error('Unhandled error:\n%s', traceback.format_exc())
    return jsonify({'error': 'Internal server error: ' + str(err)}), 500


def validate(df):
    missing = [c for c in input_features_order if c not in df.columns]
    if missing:
        raise BadRequest('Missing required column(s): ' + str(missing))
    df = df[input_features_order].copy()
    for col in numeric_features:
        values = pd.to_numeric(df[col], errors='coerce')
        if values.isna().any():
            raise BadRequest("Column '" + col + "' must be numeric")
        df[col] = values
    for col, allowed in allowed_categories.items():
        unknown = sorted(set(df[col].astype(str)) - set(allowed))
        if unknown:
            raise BadRequest("Column '" + col + "' has unknown value(s) " + str(unknown) +
                             '; allowed: ' + str(allowed))
    return df


def score(df):
    processed = pd.DataFrame(preprocessor.transform(df), columns=engineered_feature_names)
    return model.predict(processed)


@app.get('/')
def index():
    return jsonify({
        'service': 'SuperKart sales forecast API',
        'endpoints': ['/health', '/v1/predict', '/v1/predictbatch'],
        'expected_columns': input_features_order,
    })


@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'n_features': len(input_features_order)})


@app.post('/v1/predict')
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        raise BadRequest('Request body must be JSON')
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise BadRequest('JSON body must be an object or a non-empty list of objects')

    df = validate(pd.DataFrame(payload))
    preds = score(df)
    if len(preds) == 1:
        return jsonify({'prediction': float(preds[0])})
    return jsonify({'predictions': [float(p) for p in preds]})


@app.post('/v1/predictbatch')
def predict_batch():
    if 'file' not in request.files:
        raise BadRequest("No file part named 'file' in the request")
    raw = request.files['file'].read()
    if not raw:
        raise BadRequest('Uploaded file is empty')
    try:
        batch = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise BadRequest('Could not parse uploaded file as CSV: ' + str(exc))

    df = validate(batch)
    preds = score(df)
    return jsonify({str(i): float(p) for i, p in enumerate(preds)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
