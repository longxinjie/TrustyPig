import json
from pathlib import Path

metrics_path = Path("ci_outputs/metrics.json")
report_path = Path("report.md")

if not metrics_path.exists():
    raise FileNotFoundError("ci_outputs/metrics.json not found. Smoke validation must run before report generation.")

with open(metrics_path, "r") as f:
    metrics = json.load(f)

def status(value):
    return "Passed" if value else "Failed"

report = f"""# TrustyPig Pipeline CI Report

## CI Validation Checks

| Check | Status |
|---|---|
| Model file exists | {status(metrics["model_file_exists"])} |
| Model loads correctly | {status(metrics["model_loaded"])} |
| Feature schema file exists | {status(metrics["feature_schema_file_exists"])} |
| Feature schema validation | {status(metrics["feature_schema_valid"])} |
| Prediction pipeline works | {status(metrics["prediction_pipeline_valid"])} |
| Probability output range | {status(metrics["probability_range_valid"])} |
| Smoke inference test | {status(metrics["prediction_count"] == metrics["sample_count"])} |

## Smoke Test Metrics

| Metric | Value |
|---|---|
| Sample count | {metrics["sample_count"]} |
| Prediction count | {metrics["prediction_count"]} |
| Smoke test accuracy | {metrics["smoke_test_accuracy"]} |
| Minimum probability | {metrics["minimum_probability"]} |
| Maximum probability | {metrics["maximum_probability"]} |
| Confusion matrix | {metrics["confusion_matrix"]} |

## Schema Handling

| Item | Value |
|---|---|
| Missing columns filled with zero | {metrics["missing_feature_columns_filled_with_zero"]} |
| Extra sample columns ignored | {metrics["extra_sample_columns_ignored"]} |

## Confusion Matrix

"""

report_path.write_text(report)

print("CML report generated dynamically from ci_outputs/metrics.json")