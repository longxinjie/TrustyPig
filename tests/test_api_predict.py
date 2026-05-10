from app import app

def test_predict_api():
    client = app.test_client()

    payload = {
        "wallet_ratio": 0.8,
        "hour_of_day": 23,
        "amount": 1000,
        "receiver_freq": 5,
        "sender_freq": 10,
        "is_merchant": 0,
        "type_CASH_IN": 0,
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 0,
        "type_TRANSFER": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.get_json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1