"""
create_cloudwatch_dashboard.py
--------------------------------
Creates (or updates) the CloudWatch dashboard for the churn pipeline.

Run this ONCE from the EC2 instance (or any machine with AWS credentials),
after the pipeline has run at least once with CHURN_CLOUDWATCH=1 so that
the "ChurnPipeline" metrics and the "churn-pipeline-logs" log group exist:

    python create_cloudwatch_dashboard.py

Then open the printed URL to view the dashboard.
"""

import json

import boto3

DASHBOARD_NAME = "churn-pipeline-dashboard"
NAMESPACE = "ChurnPipeline"
LOG_GROUP = "churn-pipeline-logs"


def build_dashboard_body(region: str) -> str:
    def metric_widget(title, metrics, x, y, width=8, height=6, stat="Average"):
        return {
            "type": "metric", "x": x, "y": y, "width": width, "height": height,
            "properties": {
                "metrics": metrics,
                "period": 120,  # pipeline runs every 2 minutes
                "stat": stat,
                "region": region,
                "title": title,
                "view": "timeSeries",
            },
        }

    widgets = [
        metric_widget(
            "Model Accuracy over time",
            [[NAMESPACE, "accuracy", "Model", "LogisticRegression"],
             [NAMESPACE, "accuracy", "Model", "RandomForest"]],
            x=0, y=0,
        ),
        metric_widget(
            "Model F1-Score over time",
            [[NAMESPACE, "f1_score", "Model", "LogisticRegression"],
             [NAMESPACE, "f1_score", "Model", "RandomForest"]],
            x=8, y=0,
        ),
        metric_widget(
            "Model ROC-AUC over time",
            [[NAMESPACE, "roc_auc", "Model", "LogisticRegression"],
             [NAMESPACE, "roc_auc", "Model", "RandomForest"]],
            x=16, y=0,
        ),
        metric_widget(
            "Pipeline run duration (s)",
            [[NAMESPACE, "run_duration_sec"]],
            x=0, y=6,
        ),
        metric_widget(
            "Rows processed per run",
            [[NAMESPACE, "rows_processed"]],
            x=8, y=6, stat="Maximum",
        ),
        metric_widget(
            "Successful runs (per interval)",
            [[NAMESPACE, "run_success"]],
            x=16, y=6, stat="Sum",
        ),
        {
            "type": "log", "x": 0, "y": 12, "width": 24, "height": 8,
            "properties": {
                "query": (
                    f"SOURCE '{LOG_GROUP}' | fields @timestamp, @message"
                    " | sort @timestamp desc | limit 50"
                ),
                "region": region,
                "title": "Recent pipeline log events",
                "view": "table",
            },
        },
    ]
    return json.dumps({"widgets": widgets})


if __name__ == "__main__":
    session = boto3.session.Session()
    region = session.region_name or "us-east-1"

    cw = boto3.client("cloudwatch", region_name=region)
    cw.put_dashboard(
        DashboardName=DASHBOARD_NAME,
        DashboardBody=build_dashboard_body(region),
    )

    print(f"Dashboard '{DASHBOARD_NAME}' created/updated in region {region}.")
    print("Open it here:")
    print(f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
          f"#dashboards/dashboard/{DASHBOARD_NAME}")
