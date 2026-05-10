import json
from pathlib import Path

metrics_path = Path("ci_outputs/metrics.json")
report_path = Path("report.md")
confusion_matrix_path = Path("ci_outputs/confusion_matrix.png")

MIN_ACCURACY = 0.50
MIN_F1 = 0.40

if not metrics_path.exists():
    raise FileNotFoundError("ci_outputs/metrics.json not found")

with open(metrics_path, "r") as f:
    metrics = json.load(f)

def status(value):
    return "Passed" if value else "Failed"

def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return value

accuracy = metrics.get("accuracy")
f1 = metrics.get("f1")
auc = metrics.get("auc")
rows_tested = metrics.get("rows_tested")
fraud_rows = metrics.get("fraud_rows")
legit_rows = metrics.get("legit_rows")

model_loaded = metrics.get("model_loaded", False)
schema_validated = metrics.get("schema_validated", False)
prediction_pipeline_works = metrics.get("prediction_pipeline_works", False)

accuracy_passed = accuracy is not None and accuracy >= MIN_ACCURACY
f1_passed = fraud_rows == 0 or (f1 is not None and f1 >= MIN_F1)
confusion_matrix_generated = confusion_matrix_path.exists()

overall_passed = all([
    model_loaded,
    schema_validated,
    prediction_pipeline_works,
    accuracy_passed,
    f1_passed,
    confusion_matrix_generated,
])

report = f"""# TrustyPig Pipeline CI Report

## CI Validation Checks

| Check | Status |
|---|---|
| Model loads correctly | {status(model_loaded)} |
| Feature schema validation | {status(schema_validated)} |
| Prediction pipeline works | {status(prediction_pipeline_works)} |
| Smoke accuracy threshold | {status(accuracy_passed)} |
| Smoke F1 threshold | {status(f1_passed)} |
| Confusion matrix generated | {status(confusion_matrix_generated)} |

## Smoke Test Metrics

| Metric | Value |
|---|---|
| Rows tested | {fmt(rows_tested)} |
| Fraud rows | {fmt(fraud_rows)} |
| Legit rows | {fmt(legit_rows)} |
| Smoke accuracy | {fmt(accuracy)} |
| Smoke F1 | {fmt(f1)} |
| Smoke AUC | {fmt(auc)} |

## Confusion Matrix

{"![Confusion Matrix](ci_outputs/confusion_matrix.png)" if confusion_matrix_generated else "Confusion matrix was not generated."}

## CI Decision

{"The TrustyPig model passed smoke validation and is eligible for the CD stage." if overall_passed else "The TrustyPig model failed one or more CI validation checks and should not proceed to CD."}
"""

report_path.write_text(report, encoding="utf-8")

print("CML report generated dynamically from ci_outputs/metrics.json")