from clearml import Task, OutputModel
from pathlib import Path

task = Task.init(
    project_name="TrustyPig-MLOps",
    task_name="register_model_task"
)

model_version = task.get_parameter(
    "General/model_version",
    default="v1"
)

model_path = Path("model/fraud_model.pkl")

if not model_path.exists():
    raise FileNotFoundError("Model not found.")

output_model = OutputModel(
    task=task,
    name=f"trustypig-model-{model_version}",
    framework="scikit-learn"
)

output_model.update_weights(
    weights_filename=str(model_path)
)

print(f"Registered model version: {model_version}")