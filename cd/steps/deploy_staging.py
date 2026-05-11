from clearml import Task
from pathlib import Path
import json
from datetime import datetime

task = Task.init(
    project_name="TrustyPig-MLOps",
    task_name="deploy_staging_task"
)

model_version = task.get_parameter(
    "General/model_version",
    default="v1"
)

deployment_dir = Path("staging")
deployment_dir.mkdir(exist_ok=True)

manifest = {
    "environment": "staging",
    "model_version": model_version,
    "status": "deployed",
    "timestamp": str(datetime.now())
}

with open(deployment_dir / "deployment.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Deployment complete.")
print(manifest)