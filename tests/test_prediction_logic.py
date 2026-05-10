import joblib
import numpy as np

def test_model_prediction_output_shape():
    model = joblib.load("model/fraud_model.pkl")

    sample_features = np.array([[
        0.8,   # wallet_ratio
        23,    # hour_of_day
        1000,  # amount
        5,     # receiver_freq
        10,    # sender_freq
        0,     # is_merchant
        0,     # type_CASH_IN
        0,     # type_CASH_OUT
        0,     # type_DEBIT
        0,     # type_PAYMENT
        1      # type_TRANSFER
    ]])

    prediction = model.predict(sample_features)[0]
    probability = model.predict_proba(sample_features)[0][1]

    assert prediction in [0, 1]
    assert 0 <= probability <= 1