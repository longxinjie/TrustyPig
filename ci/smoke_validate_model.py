import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

MODEL_PATH = Path("fraud_model.pkl")
FEATURES_PATH = Path("feature_columns.csv")
TEST_DATA_PATH = Path("data/smoke_test_dataset.csv")

MIN_ACCURACY = 0.50
MIN_F1 = 0.40

TARGET = "isFraud"

model = joblib.load(MODEL_PATH)
feature_columns = pd.read_csv(FEATURES_PATH).iloc[:, 0].tolist()

df = pd.read_csv(TEST_DATA_PATH)

if "label" in df.columns:
    df[TARGET] = df["label"].map({"fraud": 1, "legit": 0}).fillna(df["label"]).astype(int)
elif "fraud" in df.columns:
    df[TARGET] = df["fraud"].astype(int)
elif TARGET not in df.columns:
    raise ValueError("Need label, fraud, or isFraud column for smoke validation.")

if "type" in df.columns:
    base = df["type"].astype(str).str.extract(
        r"(CASH_IN|CASH_OUT|PAYMENT|DEBIT|TRANSFER)",
        expand=False
    )

    dummies = pd.get_dummies(base, prefix="type")

    for col in [
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER"
    ]:
        if col not in dummies.columns:
            dummies[col] = 0

    df = pd.concat([df, dummies], axis=1)

missing = [c for c in feature_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required model features: {missing}")

df = df.replace([np.inf, -np.inf], np.nan)
df = df[feature_columns + [TARGET]].dropna()

X = df[feature_columns]
y = df[TARGET].astype(int)

pred = model.predict(X)

if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)[:, 1]
else:
    proba = pred

accuracy = accuracy_score(y, pred)
f1 = f1_score(y, pred)

print(f"Smoke accuracy: {accuracy:.4f}")
print(f"Smoke F1: {f1:.4f}")

if len(set(y)) > 1:
    auc = roc_auc_score(y, proba)
    print(f"Smoke AUC: {auc:.4f}")

if accuracy < MIN_ACCURACY:
    raise RuntimeError(f"Smoke accuracy too low: {accuracy:.4f}")


if f1 < MIN_F1:
    raise RuntimeError(f"Smoke F1 too low: {f1:.4f}")

print("Smoke validation passed.")

Path("ci_outputs").mkdir(exist_ok=True)

metrics = {
    "accuracy": float(accuracy),
    "f1": float(f1)
}

with open("ci_outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved CI metrics.")