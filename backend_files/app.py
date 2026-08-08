# SuperKart sales-forecasting API.
#
# Fixes applied compared with the previous version:
#  * engineered feature names come from preprocessor.get_feature_names_out()
#    instead of being rebuilt here from a second, duplicated set of hardcoded
#    column lists that could silently drift away from training;
#  * unknown or misspelt category values are rejected with an HTTP 400 naming
#    the column and the offending value. The encoder previously used
#    handle_unknown='ignore', so "Small" instead of "Low" became all zeros and
#    still returned a confident-looking number;
#  * caller mistakes return 400 while genuine server faults return 500, and the
#    traceback is logged server-side instead of being swallowed entirely;
#  * Store_Id is no longer a model input, so Batch_Data_SuperKart.csv validates
#    as-is and no fabricated default store has to be injected.
import io
import logging
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
                                                                                                    raise BadRequest("Column '" + col + "' has unknown value(s) " + str(unknown) + '; allowed: ' + str(allowed))
                                                                                                        return df


                                                                                                        def score(df):
                                                                                                            processed = pd.DataFrame(preprocessor.transform(df), columns=engineered_feature_names)
                                                                                                                return model.predict(processed)
