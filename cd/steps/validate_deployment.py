from clearml import Task
from pathlib import Path
import json

task = Task.init(
    project_name="TrustyPig-MLOps",
    task_name="validate_deployment_task"
)

model_version = task.get_parameter(
    "General/model_version",
    default="v1"
)

deployment_file = Path("staging/deployment.json")

if not deployment_file.exists():
    raise FileNotFoundError("Deployment file missing.")

with open(deployment_file, "r") as f:
    deployment = json.load(f)

assert deployment["status"] == "deployed"
assert deployment["model_version"] == model_version

print("Deployment validation successful.")