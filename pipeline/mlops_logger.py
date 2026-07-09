"""
pipeline/mlops_logger.py
--------------------------
MLOps monitoring: logs model metrics (accuracy, precision, recall, F1, ROC-AUC)
with a timestamp on every run, so metrics can be tracked/compared over time.

- Always writes to logs/model_metrics.csv (works with zero extra installs).
- If MLflow is installed (`pip install mlflow`), it ALSO logs to MLflow so you
  get the full MLflow UI dashboard (`mlflow ui` -> http://127.0.0.1:5000).
"""

import csv
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger("churn_pipeline")

METRICS_CSV = "logs/model_metrics.csv"

# Cloud mode: when CHURN_CLOUDWATCH=1, metrics are also pushed to AWS
# CloudWatch (namespace "ChurnPipeline") so they can be graphed on a dashboard.
CLOUDWATCH_ENABLED = os.environ.get("CHURN_CLOUDWATCH") == "1"
CW_NAMESPACE = "ChurnPipeline"

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _push_model_metrics_cloudwatch(metrics: dict):
    """Pushes one model's evaluation metrics to CloudWatch as custom metrics."""
    if not CLOUDWATCH_ENABLED:
        return
    try:
        import boto3
        cw = boto3.client("cloudwatch")
        data = [
            {
                "MetricName": key,
                "Value": float(value),
                "Dimensions": [{"Name": "Model", "Value": metrics["model"]}],
            }
            for key, value in metrics.items()
            if key not in ("model", "timestamp")
        ]
        cw.put_metric_data(Namespace=CW_NAMESPACE, MetricData=data)
        logger.info(f"MLOPS: model metrics pushed to CloudWatch for '{metrics['model']}'")
    except Exception as e:
        logger.warning(f"MLOPS: could not push model metrics to CloudWatch: {e}")


def push_run_metrics(duration_sec: float, rows_processed: int, success: bool):
    """Pushes run-level pipeline metrics (duration, rows, success) to CloudWatch."""
    if not CLOUDWATCH_ENABLED:
        return
    try:
        import boto3
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace=CW_NAMESPACE,
            MetricData=[
                {"MetricName": "run_duration_sec", "Value": float(duration_sec), "Unit": "Seconds"},
                {"MetricName": "rows_processed", "Value": float(rows_processed), "Unit": "Count"},
                {"MetricName": "run_success", "Value": 1.0 if success else 0.0, "Unit": "Count"},
            ],
        )
        logger.info("MLOPS: run-level metrics pushed to CloudWatch")
    except Exception as e:
        logger.warning(f"MLOPS: could not push run metrics to CloudWatch: {e}")


def log_run(metrics: dict):
    """Append one row of metrics (with timestamp) to logs/model_metrics.csv,
    and to MLflow if available."""
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), **metrics}

    file_exists = os.path.isfile(METRICS_CSV)
    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    with open(METRICS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"MLOPS: metrics logged to {METRICS_CSV} -> {row}")

    _push_model_metrics_cloudwatch(metrics)

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("telco_churn_prediction")
        with mlflow.start_run(run_name=metrics["model"]):
            mlflow.log_param("model_type", metrics["model"])
            for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
                mlflow.log_metric(key, metrics[key])
        logger.info(f"MLOPS: metrics also logged to MLflow for run '{metrics['model']}'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    sample = {"model": "TestModel", "accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1_score": 0.65, "roc_auc": 0.82}
    log_run(sample)
    print(f"Logged sample metrics to {METRICS_CSV} (MLflow available: {MLFLOW_AVAILABLE})")
