import pandas as pd, numpy as np, re
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score


# --- Config ---
RAW = "data/cleaned_dataset.csv"          
NEW = "data/app_transactions.csv"        
TARGET = "isFraud"
FEATURE_COLUMNS = [
    'wallet_ratio', 'hour_of_day', 'amount',
    'receiver_freq', 'sender_freq', 'is_merchant',
    'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

# 1. Load + combine
df_orig = pd.read_csv(RAW)
df_new = pd.read_csv(NEW)

# 1) Target: label -> isFraud (fallback to 'fraud' bool if present)
if "label" in df_new.columns:
    df_new[TARGET] = df_new["label"].map({"fraud":1, "legit":0})
elif "fraud" in df_new.columns:
    df_new[TARGET] = df_new["fraud"].astype(int)
else:
    raise ValueError("Need either 'label' or 'fraud' in app CSV.")

# 2) Normalize type strings to base types then one-hot
base = df_new["type"].astype(str).str.extract(r'(CASH_IN|CASH_OUT|PAYMENT|DEBIT|TRANSFER)', expand=False)
dummies = pd.get_dummies(base, prefix="type")
for col in ["type_CASH_IN","type_CASH_OUT","type_DEBIT","type_PAYMENT","type_TRANSFER"]:
    if col not in dummies.columns:
        dummies[col] = 0
df_new = pd.concat([df_new, dummies], axis=1)

# 3) Make sure numeric features are numeric; handle infinities
num_cols = ['wallet_ratio','hour_of_day','amount','receiver_freq','sender_freq','is_merchant']
for c in num_cols:
    if c in df_new.columns:
        df_new[c] = pd.to_numeric(df_new[c], errors="coerce")

df_new.replace([np.inf, -np.inf], np.nan, inplace=True)

# 4) Keep only the exact features + target, drop unusable rows
need = FEATURE_COLUMNS + [TARGET]
missing = [c for c in need if c not in df_new.columns]
if missing:
    raise ValueError(f"Missing in app CSV after prep: {missing}")
df_new = df_new[need].dropna()

df_all = pd.concat([df_orig, df_new], ignore_index=True)

X = df_all[FEATURE_COLUMNS]
y = df_all[TARGET].astype(int)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=100
)

# 3. Imbalance handling
non_fraud = (y_train == 0).sum()
fraud = (y_train == 1).sum()
scale_pos_weight = non_fraud / fraud

# 4. Train model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.01,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=100
)
model.fit(X_train, y_train)

# 5. Evaluate
proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= 0.5).astype(int)
print("AUC:", roc_auc_score(y_test, proba))
print("F1 :", f1_score(y_test, y_pred))

# 6. Save model + features
joblib.dump(model, "fraud_model.pkl")
pd.Series(FEATURE_COLUMNS).to_csv("feature_columns.csv", index=False)
