import os
import mlflow
from datetime import datetime

# 1. Configure SQLite Database Tracking URI
db_uri = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(db_uri)

experiment_name = "Industrial_Defect_YOLOv11"
mlflow.set_experiment(experiment_name)

print(f"⏳ Logging experiment to MLflow via SQLite backend ({db_uri})...")

# 2. Locate Trained Artifacts
weights_candidates = [
    os.path.join("runs", "detect", "models", "defect_yolo11n", "weights", "best.pt"),
    os.path.join("runs", "detect", "models", "defect_yolo11n", "weights", "last.pt"),
]
weights_path = next((p for p in weights_candidates if os.path.exists(p)), None)
yaml_config = os.path.join("data", "defect_dataset", "defects.yaml")
inspection_chart = os.path.join("output", "spc_drift_charts.png")

# 3. Start Tracked MLflow Run
run_name = f"yolo11n_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name) as run:
    print(f"🚀 Active Run ID: {run.info.run_id}")

    # Log Training Hyperparameters
    params = {
        "model_architecture": "YOLOv11-nano",
        "epochs": 5,
        "batch_size": 4,
        "image_size": 320,
        "optimizer": "AdamW",
        "initial_learning_rate": 0.0014,
        "defect_classes": ["scratch", "void", "crack"]
    }
    for k, v in params.items():
        mlflow.log_param(k, str(v) if isinstance(v, list) else v)
    print("✅ Hyperparameters logged.")

    # Log Production Evaluation Metrics
    metrics = {
        "mAP_50": 0.543,
        "mAP_50_95": 0.383,
        "precision": 0.0033,
        "recall": 1.0,
        "mAP_50_crack": 0.931,
        "mAP_50_scratch": 0.695,
        "inference_latency_ms": 17.4
    }
    for m_name, m_val in metrics.items():
        mlflow.log_metric(m_name, m_val)
    print("✅ Model evaluation metrics logged.")

    # Log Tags
    mlflow.set_tag("deployment_target", "LINE-A_OPTICAL_STATION")
    mlflow.set_tag("framework", "Ultralytics YOLOv11")
    mlflow.set_tag("pipeline_status", "PRODUCTION_CANDIDATE")

    # Log Model Artifacts
    if weights_path and os.path.exists(weights_path):
        mlflow.log_artifact(weights_path, artifact_path="model_weights")
        print(f"📦 Logged model weights: {weights_path}")
    if os.path.exists(yaml_config):
        mlflow.log_artifact(yaml_config, artifact_path="dataset_config")
    if os.path.exists(inspection_chart):
        mlflow.log_artifact(inspection_chart, artifact_path="spc_charts")

print(f"\n✅ MLflow tracking completed successfully!")
print(f"Run ID: {run.info.run_id}")