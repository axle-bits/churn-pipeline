# Telco Customer Churn — Cloud-Native Data Science Pipeline

Full working implementation for AIMLCZG549 Assignment I: a Data Pipeline (ingestion,
preprocessing, EDA, DataOps), an ML Pipeline (training, evaluation, MLOps), and an
API layer — all runnable locally, and structured so each piece can be lifted onto a
cloud provider (AWS/GCP/Azure) for the final deployment.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the data

**Option A — real Kaggle dataset (recommended for your actual submission):**
1. Download from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Rename `WA_Fn-UseC_-Telco-Customer-Churn.csv` to `telco_churn_raw.csv`
3. Place it inside the `data/` folder

**Option B — instant synthetic dataset (to test the pipeline right now):**
```bash
python generate_sample_data.py
```

## 3. Run the full pipeline once

```bash
python run_pipeline.py
```

This runs: ingestion → preprocessing → EDA → model training (70/30 split) →
evaluation (5 metrics) → MLOps metric logging, and:
- Prints progress to the console
- Writes a full activity log to `logs/pipeline.log`
- Writes model metrics history to `logs/model_metrics.csv`
- Saves charts to `artifacts/*.png`
- Saves trained models to `models/*.pkl`

## 4. Automate it (DataOps — runs every 2 minutes)

```bash
python scheduler.py
```
Leave this running in a terminal — it will trigger `run_pipeline.py`'s logic
every 2 minutes and log each run. Press Ctrl+C to stop.

**Cloud version:** replace this with an Airflow DAG (`schedule_interval='*/2 * * * *'`)
or an AWS Lambda + EventBridge rule (`rate(2 minutes)`) once deployed — see comments
inside `scheduler.py`.

## 5. Start the API

```bash
uvicorn api_app:app --reload --port 8000
```
Open http://127.0.0.1:8000/docs for the interactive Swagger UI, or call directly:

```bash
curl http://127.0.0.1:8000/api/pipeline-status
curl http://127.0.0.1:8000/api/data-summary
curl http://127.0.0.1:8000/api/model-metrics
curl http://127.0.0.1:8000/api/deployment-info
curl http://127.0.0.1:8000/api/health
```

## 6. (Optional) View the MLflow dashboard

If you installed `mlflow`, every pipeline run also logs to it automatically:
```bash
mlflow ui
```
Then open http://127.0.0.1:5000

## 7. Deploying to the cloud (for the assignment's "cloud-native" requirement)

**AWS (chosen for this project): follow the step-by-step guide in
[`AWS_DEPLOYMENT.md`](AWS_DEPLOYMENT.md)** — S3 ingestion, CloudWatch
logs + metrics + dashboard, EC2 hosting, and the full screenshot checklist.
Cloud behaviour is toggled by env vars (`CHURN_S3_BUCKET`, `CHURN_CLOUDWATCH=1`),
so the same code still runs locally with no changes.

| Component        | AWS                              | GCP                        | Azure                     |
|-------------------|-----------------------------------|-----------------------------|-----------------------------|
| Data storage      | S3                                | Cloud Storage               | Blob Storage                |
| Scheduling        | Lambda + EventBridge, or Airflow (MWAA) | Cloud Scheduler + Cloud Functions, or Cloud Composer | Logic Apps / Functions + Timer trigger |
| Logging           | CloudWatch Logs                  | Cloud Logging                | Application Insights        |
| API hosting       | Elastic Beanstalk / ECS Fargate / Lambda+API Gateway | Cloud Run             | App Service                 |
| ML tracking       | MLflow on EC2 (S3 artifact store) | MLflow on Compute Engine    | MLflow on a VM              |

Swap the local file paths in `ingestion.py` / `api_app.py` for the equivalent
SDK calls (`boto3`, `google-cloud-storage`, `azure-storage-blob`) once you deploy —
the commented code blocks show exactly where.

## Project structure

```
churn_pipeline/
├── generate_sample_data.py   # synthetic data generator (optional)
├── run_pipeline.py           # orchestrates one full pipeline run
├── scheduler.py              # DataOps: runs the pipeline every 2 minutes
├── api_app.py                # FastAPI app exposing application details
├── requirements.txt
├── pipeline/
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train_evaluate.py
│   └── mlops_logger.py
├── data/                     # raw + processed CSVs land here
├── artifacts/                # EDA charts (.png) land here
├── models/                   # trained model .pkl files land here
└── logs/                     # pipeline.log and model_metrics.csv land here
```
