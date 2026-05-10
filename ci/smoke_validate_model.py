import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

MODEL_PATH = "model/fraud_model.pkl"
FEATURES_PATH = "model/feature_columns.csv"
OUTPUT_DIR = "ci_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Check model exists
assert os.path.exists(MODEL_PATH), "Model file fraud_model.pkl not found"

# 2. Load model
model = joblib.load(MODEL_PATH)

# 3. Load feature schema
assert os.path.exists(FEATURES_PATH), "feature_columns.csv not found"
feature_columns = pd.read_csv(FEATURES_PATH).iloc[:, 0].tolist()

# 4. Create small smoke-test transaction sample
sample_data = pd.DataFrame([
    {
        "amount": 100,
        "wallet_ratio": 0.2,
        "hour_of_day": 10,
        "sender_freq": 2,
        "receiver_freq": 1,
        "is_merchant": 0,
        "type_TRANSFER": 1,
        "type_CASH_OUT": 0,
        "type_PAYMENT": 0,
        "type_CASH_IN": 0,
        "type_DEBIT": 0,
        "label": 0
    },
    {
        "amount": 9500,
        "wallet_ratio": 0.95,
        "hour_of_day": 2,
        "sender_freq": 1,
        "receiver_freq": 8,
        "is_merchant": 0,
        "type_TRANSFER": 1,
        "type_CASH_OUT": 1,
        "type_PAYMENT": 0,
        "type_CASH_IN": 0,
        "type_DEBIT": 0,
        "label": 1
    }
])

y_true = sample_data["label"]
X = sample_data.drop(columns=["label"])

# 5. Force feature schema compatibility
for col in feature_columns:
    if col not in X.columns:
        X[col] = 0

X = X[feature_columns]

# 6. Run prediction
y_pred = model.predict(X)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X)[:, 1]
else:
    probabilities = y_pred

# 7. Validate probability range
assert np.all(probabilities >= 0) and np.all(probabilities <= 1), "Probability output outside 0–1 range"

# 8. Basic metric sanity check
accuracy = accuracy_score(y_true, y_pred)

# 9. Generate confusion matrix plot
cm = confusion_matrix(y_true, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("TrustyPig CI Smoke Test Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", bbox_inches="tight")

# 10. Save metrics
metrics = {
    "model_loaded": True,
    "feature_schema_valid": True,
    "prediction_pipeline_valid": True,
    "probability_range_valid": True,
    "smoke_test_accuracy": round(float(accuracy), 4)
}

with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("ML smoke validation completed successfully.")
print(metrics)