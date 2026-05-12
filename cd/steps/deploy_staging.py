from clearml import Task
from pathlib import Path
import json
import shutil
import subprocess
import os
from datetime import datetime

# ClearML deployment task
task = Task.init(
    project_name="TrustyPig-MLOps",
    task_name="deploy_staging_task"
)

# Get model version
model_version = task.get_parameter(
    "General/model_version",
    default="v1"
)

# Paths
MODEL_PATH = Path("fraud_model.pkl")
FEATURES_PATH = Path("feature_columns.csv")
APP_PATH = Path("app.py")

STAGING_DIR = Path("staging")
STAGING_MODEL_PATH = STAGING_DIR / "fraud_model.pkl"
STAGING_FEATURES_PATH = STAGING_DIR / "feature_columns.csv"
DEPLOYMENT_MANIFEST_PATH = STAGING_DIR / "deployment.json"

# Validation checks
if not MODEL_PATH.exists():
    raise FileNotFoundError("fraud_model.pkl not found. Cannot deploy.")

if not FEATURES_PATH.exists():
    raise FileNotFoundError("feature_columns.csv not found. Cannot deploy.")

if not APP_PATH.exists():
    raise FileNotFoundError("app.py not found. Cannot start Flask app.")

# Create staging folder
STAGING_DIR.mkdir(exist_ok=True)

# Copy model + feature schema into staging
shutil.copy2(MODEL_PATH, STAGING_MODEL_PATH)
shutil.copy2(FEATURES_PATH, STAGING_FEATURES_PATH)

# Create deployment manifest
manifest = {
    "environment": "staging",
    "model_version": model_version,
    "status": "deployed",
    "model_path": str(STAGING_MODEL_PATH),
    "features_path": str(STAGING_FEATURES_PATH),
    "app": str(APP_PATH),
    "timestamp": str(datetime.now())
}

with open(DEPLOYMENT_MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)

# Upload manifest to ClearML
task.upload_artifact(
    name="staging_deployment_manifest",
    artifact_object=str(DEPLOYMENT_MANIFEST_PATH)
)

task.upload_artifact(
    name="staging_model",
    artifact_object=str(STAGING_MODEL_PATH)
)

task.upload_artifact(
    name="staging_feature_columns",
    artifact_object=str(STAGING_FEATURES_PATH)
)

print("Staging deployment prepared.")
print(json.dumps(manifest, indent=2))

# Set environment variables for Flask app
env = os.environ.copy()
env["FLASK_ENV"] = "staging"
env["MODEL_PATH"] = str(STAGING_MODEL_PATH)
env["FEATURES_PATH"] = str(STAGING_FEATURES_PATH)
env["MODEL_VERSION"] = model_version

print("Starting Flask app in staging mode...")
print("Staging app should run at: http://127.0.0.1:5000")

# Start Flask app
subprocess.run(
    ["python", "app.py"],
    env=env,
    check=True
)