import json
from pathlib import Path

metrics_path = Path("ci_outputs/metrics.json")
report_path = Path("report.md")

with open(metrics_path, "r") as f:
    metrics = json.load(f)

def status(value):
    return "Passed" if value else "Failed"

model_file_exists = metrics.get("model_file_exists", metrics.get("model_loaded", False))
model_loaded = metrics.get("model_loaded", False)
feature_schema_file_exists = metrics.get("feature_schema_file_exists", metrics.get("feature_schema_valid", False))
feature_schema_valid = metrics.get("feature_schema_valid", False)
prediction_pipeline_valid = metrics.get("prediction_pipeline_valid", False)
probability_range_valid = metrics.get("probability_range_valid", False)

sample_count = metrics.get("sample_count", "N/A")
prediction_count = metrics.get("prediction_count", "N/A")

smoke_test_passed = (
    prediction_pipeline_valid
    and probability_range_valid
    and sample_count == prediction_count
)

report = f"""# TrustyPig Pipeline CI Report

## CI Validation Checks

| Check | Status |
|---|---|
| Model file exists | {status(model_file_exists)} |
| Model loads correctly | {status(model_loaded)} |
| Feature schema file exists | {status(feature_schema_file_exists)} |
| Feature schema validation | {status(feature_schema_valid)} |
| Prediction pipeline works | {status(prediction_pipeline_valid)} |
| Probability output range | {status(probability_range_valid)} |
| Smoke inference test | {status(smoke_test_passed)} |

## Smoke Test Metrics

| Metric | Value |
|---|---|
| Sample count | {sample_count} |
| Prediction count | {prediction_count} |
| Smoke test accuracy | {metrics.get("smoke_test_accuracy", "N/A")} |
| Minimum probability | {metrics.get("minimum_probability", "N/A")} |
| Maximum probability | {metrics.get("maximum_probability", "N/A")} |
| Confusion matrix | {metrics.get("confusion_matrix", "N/A")} |

## Schema Handling

| Item | Value |
|---|---|
| Missing columns filled with zero | {metrics.get("missing_feature_columns_filled_with_zero", [])} |
| Extra sample columns ignored | {metrics.get("extra_sample_columns_ignored", [])} |
"""

report_path.write_text(report, encoding="utf-8")

print("CML report generated dynamically from ci_outputs/metrics.json")