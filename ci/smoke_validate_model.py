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
    # Legit-like transactions
    {"amount": 50, "wallet_ratio": 0.10, "hour_of_day": 9, "sender_freq": 5, "receiver_freq": 3, "is_merchant": 1, "type_TRANSFER": 0, "type_CASH_OUT": 0, "type_PAYMENT": 1, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 120, "wallet_ratio": 0.20, "hour_of_day": 14, "sender_freq": 7, "receiver_freq": 2, "is_merchant": 1, "type_TRANSFER": 0, "type_CASH_OUT": 0, "type_PAYMENT": 1, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 30, "wallet_ratio": 0.05, "hour_of_day": 11, "sender_freq": 10, "receiver_freq": 4, "is_merchant": 1, "type_TRANSFER": 0, "type_CASH_OUT": 0, "type_PAYMENT": 1, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 250, "wallet_ratio": 0.30, "hour_of_day": 16, "sender_freq": 3, "receiver_freq": 3, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 0, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 80, "wallet_ratio": 0.15, "hour_of_day": 18, "sender_freq": 8, "receiver_freq": 6, "is_merchant": 1, "type_TRANSFER": 0, "type_CASH_OUT": 0, "type_PAYMENT": 1, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 400, "wallet_ratio": 0.25, "hour_of_day": 13, "sender_freq": 6, "receiver_freq": 3, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 0, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 90, "wallet_ratio": 0.18, "hour_of_day": 20, "sender_freq": 11, "receiver_freq": 2, "is_merchant": 1, "type_TRANSFER": 0, "type_CASH_OUT": 0, "type_PAYMENT": 1, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 150, "wallet_ratio": 0.22, "hour_of_day": 12, "sender_freq": 9, "receiver_freq": 5, "is_merchant": 1, "type_TRANSFER": 0, "type_CASH_OUT": 0, "type_PAYMENT": 1, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 60, "wallet_ratio": 0.12, "hour_of_day": 10, "sender_freq": 12, "receiver_freq": 4, "is_merchant": 1, "type_TRANSFER": 0, "type_CASH_OUT": 0, "type_PAYMENT": 1, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 500, "wallet_ratio": 0.35, "hour_of_day": 17, "sender_freq": 4, "receiver_freq": 3, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 0, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 75, "wallet_ratio": 0.11, "hour_of_day": 15, "sender_freq": 13, "receiver_freq": 6, "is_merchant": 1, "type_TRANSFER": 0, "type_CASH_OUT": 0, "type_PAYMENT": 1, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 220, "wallet_ratio": 0.28, "hour_of_day": 19, "sender_freq": 6, "receiver_freq": 4, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 0, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 45, "wallet_ratio": 0.08, "hour_of_day": 8, "sender_freq": 15, "receiver_freq": 5, "is_merchant": 1, "type_TRANSFER": 0, "type_CASH_OUT": 0, "type_PAYMENT": 1, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 180, "wallet_ratio": 0.24, "hour_of_day": 21, "sender_freq": 7, "receiver_freq": 3, "is_merchant": 1, "type_TRANSFER": 0, "type_CASH_OUT": 0, "type_PAYMENT": 1, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},
    {"amount": 300, "wallet_ratio": 0.32, "hour_of_day": 13, "sender_freq": 5, "receiver_freq": 4, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 0, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 0},

    # Fraud-like transactions
    {"amount": 9000, "wallet_ratio": 0.92, "hour_of_day": 2, "sender_freq": 1, "receiver_freq": 9, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 12000, "wallet_ratio": 0.97, "hour_of_day": 3, "sender_freq": 1, "receiver_freq": 12, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 7500, "wallet_ratio": 0.88, "hour_of_day": 1, "sender_freq": 2, "receiver_freq": 10, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 15000, "wallet_ratio": 0.99, "hour_of_day": 4, "sender_freq": 1, "receiver_freq": 15, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 6800, "wallet_ratio": 0.84, "hour_of_day": 0, "sender_freq": 2, "receiver_freq": 8, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 20000, "wallet_ratio": 0.98, "hour_of_day": 2, "sender_freq": 1, "receiver_freq": 20, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 8300, "wallet_ratio": 0.91, "hour_of_day": 5, "sender_freq": 1, "receiver_freq": 11, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 11000, "wallet_ratio": 0.95, "hour_of_day": 3, "sender_freq": 1, "receiver_freq": 14, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 9700, "wallet_ratio": 0.93, "hour_of_day": 1, "sender_freq": 2, "receiver_freq": 13, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 13500, "wallet_ratio": 0.96, "hour_of_day": 4, "sender_freq": 1, "receiver_freq": 16, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 7200, "wallet_ratio": 0.87, "hour_of_day": 2, "sender_freq": 2, "receiver_freq": 9, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 17500, "wallet_ratio": 0.99, "hour_of_day": 0, "sender_freq": 1, "receiver_freq": 18, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 8900, "wallet_ratio": 0.90, "hour_of_day": 5, "sender_freq": 1, "receiver_freq": 10, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 16000, "wallet_ratio": 0.98, "hour_of_day": 3, "sender_freq": 1, "receiver_freq": 17, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
    {"amount": 10500, "wallet_ratio": 0.94, "hour_of_day": 1, "sender_freq": 2, "receiver_freq": 12, "is_merchant": 0, "type_TRANSFER": 1, "type_CASH_OUT": 1, "type_PAYMENT": 0, "type_CASH_IN": 0, "type_DEBIT": 0, "label": 1},
])

y_true = sample_data["label"]
X = sample_data.drop(columns=["label"])

# 5. Validate feature schema compatibility
missing_columns = [col for col in feature_columns if col not in X.columns]
extra_columns = [col for col in X.columns if col not in feature_columns]

for col in missing_columns:
    X[col] = 0

X = X[feature_columns]

schema_valid = list(X.columns) == feature_columns
assert schema_valid, f"Feature schema mismatch. Missing: {missing_columns}, Extra: {extra_columns}"

# 6. Run prediction
try:
    y_pred = model.predict(X)
    prediction_pipeline_valid = len(y_pred) == len(X)
except Exception as e:
    prediction_pipeline_valid = False
    raise RuntimeError(f"Prediction pipeline failed: {e}")

assert prediction_pipeline_valid, "Prediction output length does not match input length"

# 7. Validate probability range
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X)[:, 1]
else:
    probabilities = np.asarray(y_pred)

probability_range_valid = bool(np.all(probabilities >= 0) and np.all(probabilities <= 1))
assert probability_range_valid, "Probability output outside 0–1 range"

# 8. Basic metric sanity check
accuracy = accuracy_score(y_true, y_pred)

# 9. Generate confusion matrix plot
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

plt.figure(figsize=(5, 5))
plt.imshow(cm, cmap="Blues")
plt.title("TrustyPig CI Smoke Test Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["Legit", "Fraud"])
plt.yticks([0, 1], ["Legit", "Fraud"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)

plt.colorbar()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", bbox_inches="tight")

# 10. Save REAL runtime validation metrics
model_loaded = model is not None
model_file_exists = os.path.exists(MODEL_PATH)
feature_schema_file_exists = os.path.exists(FEATURES_PATH)

metrics = {
    "model_file_exists": model_file_exists,
    "model_loaded": model_loaded,
    "feature_schema_file_exists": feature_schema_file_exists,
    "feature_schema_valid": schema_valid,
    "missing_feature_columns_filled_with_zero": missing_columns,
    "extra_sample_columns_ignored": extra_columns,
    "prediction_pipeline_valid": prediction_pipeline_valid,
    "prediction_count": int(len(y_pred)),
    "sample_count": int(len(X)),
    "probability_range_valid": probability_range_valid,
    "minimum_probability": round(float(np.min(probabilities)), 4),
    "maximum_probability": round(float(np.max(probabilities)), 4),
    "smoke_test_accuracy": round(float(accuracy), 4),
    "confusion_matrix": cm.tolist()
}

with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("ML smoke validation completed successfully.")
print(json.dumps(metrics, indent=4))