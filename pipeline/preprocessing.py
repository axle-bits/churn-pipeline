"""
pipeline/preprocessing.py
---------------------------
Data pre-processing:
  - summary statistics
  - data type inspection
  - missing value detection
  - numeric imputation (median)
  - normalization (MinMaxScaler)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("churn_pipeline")

PROCESSED_PATH = "data/telco_churn_preprocessed.csv"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- 1. Summary statistics ----
    logger.info("PREPROCESSING: summary statistics computed")
    summary = df.describe(include="all")

    # ---- 2. Data types ----
    logger.info(f"PREPROCESSING: dtypes=\n{df.dtypes.to_string()}")

    # ---- 3. Fix TotalCharges (Kaggle quirk: blank strings for tenure=0) ----
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # ---- 4. Missing value detection ----
    missing_before = df.isnull().sum()
    missing_report = missing_before[missing_before > 0]
    logger.info(f"PREPROCESSING: missing values before imputation:\n{missing_report.to_string()}")

    # ---- 5. Numeric imputation (median) ----
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"PREPROCESSING: imputed '{col}' with median={median_val:.2f}")

    assert df.isnull().sum().sum() == 0, "There are still missing values after imputation!"

    # ---- 6. Normalization ----
    scale_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    logger.info(f"PREPROCESSING: normalized columns={scale_cols}")

    df.to_csv(PROCESSED_PATH, index=False)
    logger.info(f"PREPROCESSING: saved to {PROCESSED_PATH} | shape={df.shape}")

    return df, summary


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from project root
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from ingestion import ingest_data

    raw_df = ingest_data()
    processed_df, summary_stats = preprocess(raw_df)

    print("\n--- Summary statistics (first 5 columns) ---")
    print(summary_stats.iloc[:, :5])
    print("\n--- Processed data head ---")
    print(processed_df.head())
