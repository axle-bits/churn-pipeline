"""
api_app.py
------------
API Access requirement: exposes at least 4 key application details via REST API.

Run with:   uvicorn api_app:app --reload --port 8000
Then open:  http://127.0.0.1:8000/docs   (interactive Swagger UI, auto-generated)

Endpoints:
  GET /api/pipeline-status     -> last pipeline run status, duration, timestamp (from logs/pipeline.log)
  GET /api/data-summary        -> shape/size/last-modified of the latest processed dataset
  GET /api/model-metrics       -> most recent metrics for both trained models (from logs/model_metrics.csv)
  GET /api/deployment-info     -> runtime/environment details of this running application
  GET /api/health              -> simple health check
  GET /                        -> friendly landing page

CLOUD DEPLOYMENT: once working locally, deploy this same app to:
  - AWS: Elastic Beanstalk / ECS Fargate / Lambda+API Gateway (via Mangum)
  - GCP: Cloud Run (`gcloud run deploy`)
  - Azure: App Service
Then /api/deployment-info and /api/pipeline-status can be extended to call
boto3 / Airflow's REST API / MLflow's REST API for live cloud resource details
(see the commented example inside deployment_info() below).
"""

import csv
import os
import platform
import re
import sys
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Churn Pipeline Application API",
    description="Exposes data pipeline, model, and deployment details for the "
                 "Telco Customer Churn Prediction cloud application.",
    version="1.0.0",
)

LOG_FILE = "logs/pipeline.log"
METRICS_CSV = "logs/model_metrics.csv"
PROCESSED_DATA = "data/telco_churn_eda.csv"


@app.get("/")
def home():
    return {
        "message": "Churn Pipeline Application API is running.",
        "docs": "/docs",
        "endpoints": [
            "/api/pipeline-status", "/api/data-summary",
            "/api/model-metrics", "/api/deployment-info", "/api/health",
        ],
    }


@app.get("/api/pipeline-status")
def pipeline_status():
    """Detail 1: status/timestamp/duration of the most recent pipeline run."""
    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="No pipeline runs found yet. Run run_pipeline.py first.")

    with open(LOG_FILE) as f:
        lines = f.readlines()

    completed = [l for l in lines if "PIPELINE RUN COMPLETED" in l or "PIPELINE RUN FAILED" in l]
    if not completed:
        raise HTTPException(status_code=404, detail="No completed pipeline runs found in the log yet.")

    last_line = completed[-1]
    timestamp = last_line.split(" | ")[0]
    status = "success" if "COMPLETED" in last_line else "failed"
    duration_match = re.search(r"in ([\d.]+)s", last_line)
    duration = float(duration_match.group(1)) if duration_match else None

    return {
        "last_run_timestamp": timestamp,
        "status": status,
        "duration_sec": duration,
        "total_runs_logged": len(completed),
    }


@app.get("/api/data-summary")
def data_summary():
    """Detail 2: shape and freshness of the latest processed dataset."""
    if not os.path.exists(PROCESSED_DATA):
        raise HTTPException(status_code=404, detail="No processed dataset found yet. Run the pipeline first.")

    df = pd.read_csv(PROCESSED_DATA)
    stat = os.stat(PROCESSED_DATA)
    return {
        "file": PROCESSED_DATA,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "file_size_bytes": stat.st_size,
        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


@app.get("/api/model-metrics")
def model_metrics():
    """Detail 3: most recent evaluation metrics for each trained model."""
    if not os.path.exists(METRICS_CSV):
        raise HTTPException(status_code=404, detail="No model metrics found yet. Run the pipeline first.")

    rows = []
    with open(METRICS_CSV) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise HTTPException(status_code=404, detail="Metrics file is empty.")

    latest_by_model = {}
    for row in rows:
        latest_by_model[row["model"]] = row  # keeps the last occurrence per model

    return {"latest_metrics": list(latest_by_model.values()), "total_logged_runs": len(rows)}


def _ec2_instance_details():
    """Queries the EC2 Instance Metadata Service (IMDSv2) - AWS's built-in API
    that every EC2 instance exposes - for live deployment details."""
    import urllib.request

    base = "http://169.254.169.254/latest"
    token_req = urllib.request.Request(
        f"{base}/api/token", method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    token = urllib.request.urlopen(token_req, timeout=1).read().decode()

    def get(item):
        req = urllib.request.Request(
            f"{base}/meta-data/{item}",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(req, timeout=1).read().decode()

    return {
        "cloud_provider": "AWS EC2",
        "instance_id": get("instance-id"),
        "instance_type": get("instance-type"),
        "region": get("placement/region"),
        "availability_zone": get("placement/availability-zone"),
        "public_ip": get("public-ipv4"),
        "ami_id": get("ami-id"),
    }


@app.get("/api/deployment-info")
def deployment_info():
    """Detail 4: runtime/deployment details of this running application.

    On AWS this returns live EC2 instance details fetched from the instance
    metadata API; when run locally it falls back to local machine details.
    """
    try:
        cloud = _ec2_instance_details()
    except Exception:
        cloud = {"cloud_provider": "local (not deployed on EC2)"}

    return {
        "application": "Churn Pipeline API",
        **cloud,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_id": os.getpid(),
        "working_directory": os.getcwd(),
        "server_time_utc": datetime.utcnow().isoformat(),
    }


@app.get("/api/health")
def health():
    """Detail 5 (bonus): simple health/liveness check."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
