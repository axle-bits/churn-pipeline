"""
run_pipeline.py
------------------
Orchestrates ONE full run of the pipeline:
    ingest -> preprocess -> EDA -> train -> evaluate -> log metrics

Run this manually with:  python run_pipeline.py
Or let scheduler.py call run_once() automatically every 2 minutes.
"""

import logging
import os
import time

os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("churn_pipeline")

# Keep AWS SDK internals (credential/IAM lookups, HTTP retries) out of our logs
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))

from ingestion import ingest_data
from preprocessing import preprocess
from eda import run_eda
from train_evaluate import load_features, train_models, evaluate_model
from mlops_logger import log_run, push_run_metrics

# Cloud mode: when CHURN_CLOUDWATCH=1, every log line is ALSO streamed to an
# AWS CloudWatch log group so the runs can be monitored on a cloud dashboard.
# NOTE: this handler must be attached AFTER the pipeline imports above -
# importing mlflow reconfigures Python logging and closes any handlers
# attached before it, which would silently kill the CloudWatch stream.
CLOUDWATCH_ENABLED = os.environ.get("CHURN_CLOUDWATCH") == "1"
if CLOUDWATCH_ENABLED:
    try:
        import watchtower
        cw_handler = watchtower.CloudWatchLogHandler(
            log_group_name="churn-pipeline-logs", log_stream_name="pipeline-runs"
        )
        cw_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logging.getLogger().addHandler(cw_handler)
        logger.info("CloudWatch log streaming enabled (log group: churn-pipeline-logs)")
    except Exception as e:
        logger.warning(f"Could not enable CloudWatch logging, continuing with local logs only: {e}")


def run_once():
    start = time.time()
    logger.info("=" * 70)
    logger.info("PIPELINE RUN STARTED")
    try:
        raw_df = ingest_data()
        processed_df, _ = preprocess(raw_df)
        eda_df = run_eda(processed_df)

        X, y = load_features()
        log_reg, rf_clf, X_train, X_test, y_train, y_test = train_models(X, y)

        metrics_lr = evaluate_model(log_reg, X_test, y_test, "LogisticRegression")
        metrics_rf = evaluate_model(rf_clf, X_test, y_test, "RandomForest")
        log_run(metrics_lr)
        log_run(metrics_rf)

        duration = round(time.time() - start, 2)
        push_run_metrics(duration, raw_df.shape[0], success=True)
        logger.info(f"PIPELINE RUN COMPLETED SUCCESSFULLY in {duration}s")
        logger.info("=" * 70)
        return {"status": "success", "duration_sec": duration, "metrics": [metrics_lr, metrics_rf]}

    except Exception as e:
        logger.exception(f"PIPELINE RUN FAILED: {e}")
        push_run_metrics(round(time.time() - start, 2), 0, success=False)
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    result = run_once()
    print("\nRun result:", result)
