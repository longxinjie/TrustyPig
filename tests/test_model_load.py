import os
import joblib

def test_model_file_exists():
    assert os.path.exists("model/fraud_model.pkl")

def test_model_can_load():
    model = joblib.load("model/fraud_model.pkl")
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")