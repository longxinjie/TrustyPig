import os
import json
import shutil
from datetime import datetime

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

PACKAGE_DIR = "ci_package"

def package_artifacts():
    if os.path.exists(PACKAGE_DIR):
        shutil.rmtree(PACKAGE_DIR)

    os.makedirs(PACKAGE_DIR, exist_ok=True)

    shutil.copy("app.py", PACKAGE_DIR)
    shutil.copy("requirements.txt", PACKAGE_DIR)

    os.makedirs(f"{PACKAGE_DIR}/model", exist_ok=True)
    shutil.copy("model/fraud_model.pkl", f"{PACKAGE_DIR}/model/fraud_model.pkl")

    if os.path.exists("model/feature_columns.csv"):
        shutil.copy("model/feature_columns.csv", f"{PACKAGE_DIR}/model/feature_columns.csv")

    with open(f"{PACKAGE_DIR}/build_info.txt", "w") as f:
        f.write(f"TrustyPig CI Package\n")
        f.write(f"Build time: {datetime.now()}\n")
        f.write(f"Included: app.py, requirements.txt, model artifacts\n")
    
    manifest = {
    "model_name": "TrustyPig Fraud Detection",
    "model_version": MODEL_VERSION,
    "build_time": str(datetime.now()),
    "pipeline_stage": "CI",
    "artifacts": [
        "app.py",
        "requirements.txt",
        "fraud_model.pkl",
        "feature_columns.csv"
        ]
    }

    with open(f"{PACKAGE_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("CI package created successfully.")

if __name__ == "__main__":
    package_artifacts()