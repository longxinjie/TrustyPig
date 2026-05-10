import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

MODEL_PATH = Path("fraud_model.pkl")
FEATURES_PATH = Path("feature_columns.csv")
TEST_DATA_PATH = Path("data/smoke_test_dataset.csv")

OUTPUT_DIR = Path("ci_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_ACCURACY = 0.50
MIN_F1 = 0.40

TARGET = "isFraud"

# 1. Load model + feature schema
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")

if not TEST_DATA_PATH.exists():
    raise FileNotFoundError(f"Smoke test dataset not found: {TEST_DATA_PATH}")

model = joblib.load(MODEL_PATH)
feature_columns = pd.read_csv(FEATURES_PATH).iloc[:, 0].tolist()

# 2. Load smoke test data
df = pd.read_csv(TEST_DATA_PATH)

# 3. Convert label column to isFraud
if "label" in df.columns:
    df[TARGET] = (
        df["label"]
        .map({"fraud": 1, "legit": 0})
        .fillna(df["label"])
        .astype(int)
    )
elif "fraud" in df.columns:
    df[TARGET] = df["fraud"].astype(int)
elif TARGET in df.columns:
    df[TARGET] = df[TARGET].astype(int)
else:
    raise ValueError("Need label, fraud, or isFraud column for smoke validation.")

# 4. One-hot encode transaction type if raw type column exists
if "type" in df.columns:
    base = df["type"].astype(str).str.extract(
        r"(CASH_IN|CASH_OUT|PAYMENT|DEBIT|TRANSFER)",
        expand=False,
    )

    dummies = pd.get_dummies(base, prefix="type")

    for col in [
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ]:
        if col not in dummies.columns:
            dummies[col] = 0

    df = pd.concat([df, dummies], axis=1)

# 5. Check required features
missing = [c for c in feature_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required model features: {missing}")

# 6. Clean data
df = df.replace([np.inf, -np.inf], np.nan)
df = df[feature_columns + [TARGET]].dropna()

if df.empty:
    raise ValueError("Smoke test dataset is empty after cleaning.")

X = df[feature_columns]
y = df[TARGET].astype(int)

# 7. Predict
pred = model.predict(X)

if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)[:, 1]
else:
    proba = pred

# 8. Metrics
accuracy = accuracy_score(y, pred)
f1 = f1_score(y, pred, zero_division=0)

metrics = {
    "model_loaded": True,
    "schema_validated": True,
    "prediction_pipeline_works": True,
    "accuracy": float(accuracy),
    "f1": float(f1),
    "rows_tested": int(len(df)),
    "fraud_rows": int(y.sum()),
    "legit_rows": int((y == 0).sum()),
}

print(f"Smoke accuracy: {accuracy:.4f}")
print(f"Smoke F1: {f1:.4f}")
print(f"Rows tested: {len(df)}")
print(f"Fraud rows: {y.sum()}")
print(f"Legit rows: {(y == 0).sum()}")

if len(set(y)) > 1:
    auc = roc_auc_score(y, proba)
    metrics["auc"] = float(auc)
    print(f"Smoke AUC: {auc:.4f}")
else:
    print("Skipping AUC because smoke dataset has only one class.")

# 9. Save metrics for CML
with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 10. Save confusion matrix image for CML
cm = confusion_matrix(y, pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legit", "Fraud"],
)

disp.plot()
plt.title("TrustyPig Fraud Detection Confusion Matrix")
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", bbox_inches="tight")
plt.close()

print("Saved ci_outputs/metrics.json")
print("Saved ci_outputs/confusion_matrix.png")

# 11. CI gate checks
if accuracy < MIN_ACCURACY:
    raise RuntimeError(f"Smoke accuracy too low: {accuracy:.4f}")


if f1 < MIN_F1:
    raise RuntimeError(f"Smoke F1 too low: {f1:.4f}")


print("Smoke validation passed.")