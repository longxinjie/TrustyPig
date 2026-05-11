from clearml import PipelineController
import os

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

pipe = PipelineController(
    project="TrustyPig-MLOps",
    name=f"Fraud Detection CD Pipeline - {MODEL_VERSION}",
    version=MODEL_VERSION,
    add_pipeline_tags=True
)

# execution queue
pipe.set_default_execution_queue("default")

# STEP 1 — Register model
pipe.add_step(
    name="register_model",
    base_task_project="TrustyPig-MLOps",
    base_task_name="register_model_task",
    parameter_override={
        "General/model_version": MODEL_VERSION
    }
)

# STEP 2 — Deploy to staging
pipe.add_step(
    name="deploy_staging",
    parents=["register_model"],
    base_task_project="TrustyPig-MLOps",
    base_task_name="deploy_staging_task",
    parameter_override={
        "General/model_version": MODEL_VERSION
    }
)

# STEP 3 — Validate deployment
pipe.add_step(
    name="validate_deployment",
    parents=["deploy_staging"],
    base_task_project="TrustyPig-MLOps",
    base_task_name="validate_deployment_task",
    parameter_override={
        "General/model_version": MODEL_VERSION
    }
)

pipe.start()
print("CD Pipeline started.")