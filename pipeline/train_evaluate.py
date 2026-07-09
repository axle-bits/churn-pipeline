"""
pipeline/train_evaluate.py
----------------------------
Model preparation, training (70/30 split), and evaluation for two algorithms:
  1. Logistic Regression
  2. Random Forest Classifier

Also logs >= 4 metrics per model (accuracy, precision, recall, F1, ROC-AUC)
via mlops_logger.py (MLOps monitoring requirement).
"""

import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger("churn_pipeline")

DATA_PATH = "data/telco_churn_eda.csv"
MODEL_DIR = "models"


def load_features(path: str = DATA_PATH):
    df = pd.read_csv(path)
    y = df["Churn_enc"]
    drop_cols = ["Churn", "Churn_enc", "customerID", "tenure_group", "gender", "Partner", "Dependents"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number, bool]).astype(float)
    return X, y


def train_models(X, y, test_size=0.30, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"TRAINING: split complete | train={X_train.shape[0]} ({100*(1-test_size):.0f}%) "
        f"| test={X_test.shape[0]} ({100*test_size:.0f}%)"
    )

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    logger.info("TRAINING: LogisticRegression fit complete")

    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=random_state)
    rf_clf.fit(X_train, y_train)
    logger.info("TRAINING: RandomForestClassifier fit complete")

    joblib.dump(log_reg, f"{MODEL_DIR}/logistic_regression.pkl")
    joblib.dump(rf_clf, f"{MODEL_DIR}/random_forest.pkl")
    logger.info(f"TRAINING: models saved to {MODEL_DIR}/")

    return log_reg, rf_clf, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }
    cm = confusion_matrix(y_test, preds)
    logger.info(f"EVALUATION [{name}]: {metrics}")
    logger.info(f"EVALUATION [{name}]: confusion matrix=\n{cm}")
    return metrics


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    from mlops_logger import log_run

    X, y = load_features()
    log_reg, rf_clf, X_train, X_test, y_train, y_test = train_models(X, y)

    metrics_lr = evaluate_model(log_reg, X_test, y_test, "LogisticRegression")
    metrics_rf = evaluate_model(rf_clf, X_test, y_test, "RandomForest")

    log_run(metrics_lr)
    log_run(metrics_rf)

    print("\n=== Model Comparison ===")
    print(pd.DataFrame([metrics_lr, metrics_rf]).set_index("model"))
