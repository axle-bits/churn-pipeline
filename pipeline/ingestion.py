"""
pipeline/ingestion.py
----------------------
Data ingestion step.

Local mode (default): reads data/telco_churn_raw.csv directly.
Cloud mode: set the environment variable CHURN_S3_BUCKET to your S3 bucket
name and the file is pulled from S3 on every run, so the pipeline always
works on the latest uploaded data.
"""

import logging
import os
import pandas as pd

RAW_PATH = "data/telco_churn_raw.csv"

S3_BUCKET = os.environ.get("CHURN_S3_BUCKET")            # e.g. "churn-pipeline-raw-group43"
S3_KEY = os.environ.get("CHURN_S3_KEY", "telco_churn_raw.csv")

logger = logging.getLogger("churn_pipeline")


def ingest_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Reads the raw customer data and returns a DataFrame."""
    try:
        if S3_BUCKET:
            import boto3
            s3 = boto3.client("s3")
            s3.download_file(S3_BUCKET, S3_KEY, path)
            logger.info(f"INGESTION: downloaded s3://{S3_BUCKET}/{S3_KEY} -> {path}")

        df = pd.read_csv(path)
        logger.info(f"INGESTION: success | rows={df.shape[0]} cols={df.shape[1]} | source={path}")
        return df
    except Exception as e:
        logger.error(f"INGESTION: failed | error={e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    df = ingest_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
