# AWS Deployment Guide — Churn Pipeline (Group 43)

Target architecture (all free-tier eligible):

```
Kaggle CSV ──> S3 bucket (raw data)
                  │
                  ▼  every 2 minutes (scheduler.py)
            EC2 instance ── ingest -> preprocess -> EDA -> train -> evaluate
                  │                                   │
                  │  logs (watchtower)                │  metrics (boto3)
                  ▼                                   ▼
        CloudWatch Logs  ──────────────>  CloudWatch Dashboard
                  │
            MLflow UI (port 5000)   FastAPI app (port 8000)
```

The same code runs locally and on AWS. Cloud behaviour is switched on with
two environment variables:

| Env var            | Effect                                                        |
|--------------------|---------------------------------------------------------------|
| `CHURN_S3_BUCKET`  | ingestion pulls the raw CSV from this S3 bucket on every run  |
| `CHURN_CLOUDWATCH=1` | logs stream to CloudWatch Logs + metrics push to CloudWatch |

---

## Step 1 — Create the S3 bucket and upload the dataset

1. AWS Console → S3 → **Create bucket**
   - Name: `churn-pipeline-raw-group43` (bucket names are global; append digits if taken)
   - Region: `us-east-1` (N. Virginia) — use the SAME region for everything
   - Leave all defaults (Block Public Access ON is fine)
2. Open the bucket → **Upload** → add `data/telco_churn_raw.csv` from this repo.

📸 **Screenshot #2**: the bucket showing `telco_churn_raw.csv` uploaded (name + size visible).

## Step 2 — Create an IAM role for the EC2 instance

1. AWS Console → IAM → Roles → **Create role**
2. Trusted entity: **AWS service** → Use case: **EC2**
3. Attach permission policies:
   - `AmazonS3ReadOnlyAccess`
   - `CloudWatchFullAccess`
4. Role name: `churn-pipeline-ec2-role` → Create.

## Step 3 — Launch the EC2 instance

1. AWS Console → EC2 → **Launch instance**
   - Name: `churn-pipeline-server`
   - AMI: **Ubuntu Server 24.04 LTS**
   - Instance type: **t3.medium** (NOT free tier — ~$0.04/hour, so stop the
     instance whenever you're not actively working; t2.micro/t3.micro work
     too if you want to stay free-tier)
   - Key pair: create new (`churn-key`), download the `.pem` file, keep it safe
   - Network settings → Edit → add these **security group rules**:
     - SSH (22) — Source: *My IP*
     - Custom TCP (8000) — Source: *My IP*   ← FastAPI
     - Custom TCP (5000) — Source: *My IP*   ← MLflow UI
   - Advanced details → **IAM instance profile**: `churn-pipeline-ec2-role`
2. Launch, wait for state *Running*, note the **Public IPv4 address**.

📸 **Screenshot #11**: EC2 console showing the running instance (ID, type, public IP).

## Step 4 — Get the project onto the instance

The code lives at https://github.com/axle-bits/churn-pipeline — just clone it:

```bash
ssh -i path/to/churn-key.pem ubuntu@<PUBLIC_IP>
git clone https://github.com/axle-bits/churn-pipeline.git
```

## Step 5 — Install and configure on the instance

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv
cd ~/churn-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt        # takes a few minutes

# Cloud-mode switches (also add these lines to ~/.bashrc so they persist)
export CHURN_S3_BUCKET=churn-pipeline-raw-group43
export CHURN_CLOUDWATCH=1
export AWS_DEFAULT_REGION=us-east-1
```

## Step 6 — Run the pipeline once and verify CloudWatch

```bash
python run_pipeline.py
```

Expected in the console: `INGESTION: downloaded s3://...`,
`CloudWatch log streaming enabled`, metrics for both models, and
`PIPELINE RUN COMPLETED SUCCESSFULLY`.

Then check: AWS Console → CloudWatch → Log groups → `churn-pipeline-logs`
→ stream `pipeline-runs` should contain the run's log lines.

📸 **Screenshot #3**: SSH terminal showing the full pipeline run output (7043 rows, 70/30 split, metrics).
📸 **Screenshot #6**: CloudWatch log group `churn-pipeline-logs` with log events visible.

## Step 7 — Create the CloudWatch dashboard

```bash
python create_cloudwatch_dashboard.py
```

Open the printed URL. Widgets fill up as more scheduled runs push metrics —
after ~10 minutes of the scheduler running you'll have nice time-series lines.

📸 **Screenshot #7**: the CloudWatch dashboard with all widgets populated
(accuracy/F1/ROC-AUC lines, run duration, rows processed, recent log events).

## Step 8 — Start the scheduler (DataOps, every 2 minutes)

```bash
nohup python scheduler.py > scheduler.out 2>&1 &
```

Let it run for at least 10 minutes.

📸 **Screenshot #5**: `tail -50 logs/pipeline.log` showing 3+ consecutive
runs with timestamps exactly 2 minutes apart.

## Step 9 — Start the API and MLflow UI

```bash
nohup uvicorn api_app:app --host 0.0.0.0 --port 8000 > api.out 2>&1 &
nohup mlflow ui --host 0.0.0.0 --port 5000 > mlflow.out 2>&1 &
```

From your laptop's browser:

- `http://<PUBLIC_IP>:8000/docs` — Swagger UI
- `http://<PUBLIC_IP>:8000/api/pipeline-status`, `/api/data-summary`,
  `/api/model-metrics`, `/api/deployment-info`, `/api/health`
- `http://<PUBLIC_IP>:5000` — MLflow experiment `telco_churn_prediction`

📸 **Screenshot #8**: MLflow UI runs table showing both models with all 5 metrics.
📸 **Screenshot #9**: Swagger page listing all endpoints (URL bar showing the EC2 public IP).
📸 **Screenshot #10**: JSON responses of the 4 main endpoints (Postman collection or browser).
`/api/deployment-info` should show the real instance ID / type / region — proof of cloud deployment.

---

## Full screenshot checklist → report section mapping

Save all screenshots in the `screenshot/` folder (next to the report, NOT
inside this repo) named `ss_<number>_<short-name>.png`:

| #  | Suggested filename            | What to capture                                                | Report section |
|----|-------------------------------|----------------------------------------------------------------|----------------|
| 1  | `ss_1_kaggle.png`             | Kaggle dataset page (blastchar/telco-customer-churn)           | 2.2 Data Ingestion |
| 2  | `ss_2_s3_bucket.png`          | S3 bucket with the uploaded CSV                                | 2.2 Data Ingestion |
| 3  | `ss_3_pipeline_run.png`       | SSH terminal: full `run_pipeline.py` output                    | 2.3 Pre-processing & 3.2 Training |
| 4  | `ss_4a..4e_eda_*.png`         | The 5 charts from `artifacts/` (copy back with `scp`)          | 2.4 EDA |
| 5  | `ss_5_scheduler_log.png`      | `logs/pipeline.log` showing runs 2 minutes apart               | 2.5 DataOps |
| 6  | `ss_6_cloudwatch_logs.png`    | CloudWatch log group with events                               | 2.5.2 Logging |
| 7  | `ss_7_dashboard.png`          | CloudWatch dashboard, widgets populated                        | 2.5.3 Cloud Dashboard |
| 8  | `ss_8_mlflow.png`             | MLflow UI with both models' metrics                            | 3.4 MLOps |
| 9  | `ss_9_swagger.png`            | Swagger UI at the EC2 public IP                                | 4.1 API Access |
| 10 | `ss_10_api_responses.png`     | 4+ endpoint JSON responses (incl. deployment-info with EC2 details) | 4.2 Display Details |
| 11 | `ss_11_ec2_instance.png`      | EC2 console: running instance                                  | 5. Cloud Architecture |
| 12 | `ss_12_architecture.png`      | Architecture diagram (draw.io — redraw the diagram at the top of this file) | 5. Cloud Architecture |

### Privacy: keep AWS account details OUT of every screenshot

Before saving each console screenshot, crop or blur:

- The **account menu (top-right)** of the AWS console — shows account ID/alias
- Any **account ID** embedded in ARNs (e.g. `arn:aws:iam::123456789012:...`)
- **IAM user names / sign-in URLs**
- The **key pair filename/path** in terminal prompts is fine, but never show
  the key contents or any `AWS_ACCESS_KEY`/`AWS_SECRET` values

Instance IDs, bucket name, region, and the EC2 public IP are OK to show
(the instance is temporary and gets terminated after the demo).

To copy the charts back to your laptop:

```powershell
scp -i path\to\churn-key.pem -r ubuntu@<PUBLIC_IP>:~/churn-pipeline/artifacts .\artifacts_ec2
```

## Cost control

- t2.micro/t3.micro, 750 hrs/month free for the first 12 months; S3/CloudWatch
  usage here is well within free tier.
- **Stop the instance** (EC2 → Instance state → Stop) when you're not working
  on it; **terminate** it after the demo video is recorded and graded.
- The scheduler doesn't need to run 24/7 — 15–20 minutes of runs is plenty of
  dashboard/log evidence.
