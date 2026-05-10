from pathlib import Path
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from clearml import Task


def export_data():
    task = Task.init(
        project_name="ClearML-CICD-Demo",
        task_name="export_transaction_data",
        reuse_last_task_id=False
    )

    params = {
        "firebase_cred_path": "../smu-fintech-mvp-2025-firebase-adminsdk-fbsvc-f9c8e62f9b.json",
        "output_csv": "data/app_transactions.csv",
        "allowed_labels": ["fraud", "legit"]
    }
    task.connect(params)

    cred_path = Path(params["firebase_cred_path"])
    output_path = Path(params["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cred_path.exists():
        raise FileNotFoundError(f"Firebase credential file not found: {cred_path}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    all_txns = []
    users = db.collection("users").stream()

    for u in users:
        txns = db.collection("users").document(u.id).collection("transactions").stream()
        for t in txns:
            d = t.to_dict()
            if d.get("label") in params["allowed_labels"]:
                all_txns.append(d)

    df = pd.DataFrame(all_txns)
    df.to_csv(output_path, index=False)

    logger = task.get_logger()
    logger.report_scalar("export", "rows_exported", iteration=0, value=len(df))

    if "label" in df.columns:
        label_counts = df["label"].value_counts().to_dict()
        for label, count in label_counts.items():
            logger.report_scalar("label_counts", str(label), iteration=0, value=count)

    task.upload_artifact(
        name="app_transactions_csv",
        artifact_object=str(output_path)
    )

    print(f"Exported {len(df)} rows to {output_path}")


if __name__ == "__main__":
    export_data()