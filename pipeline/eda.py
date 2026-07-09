"""
pipeline/eda.py
-----------------
Exploratory Data Analysis:
  - correlation coefficients (numeric features)
  - categorical association (Chi-square test)
  - binning (tenure)
  - encoding (label + one-hot)
  - feature importance (Random Forest)
  - visualizations (univariate + bivariate), saved to artifacts/
"""

import logging
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("churn_pipeline")

ARTIFACT_DIR = "artifacts"
EDA_OUTPUT_PATH = "data/telco_churn_eda.csv"


def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- 1. Correlation coefficients (numeric) ----
    num_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
    corr = df[num_cols].corr()
    logger.info(f"EDA: correlation matrix:\n{corr.round(3).to_string()}")

    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap - Numeric Features")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/correlation_heatmap.png")
    plt.close()

    # ---- 2. Categorical association: Contract vs Churn (Chi-square) ----
    ct = pd.crosstab(df["Contract"], df["Churn"])
    chi2, pval, dof, _ = chi2_contingency(ct)
    logger.info(f"EDA: Chi-square(Contract vs Churn) = {chi2:.2f}, p-value = {pval:.6f}")

    # ---- 3. Binning tenure ----
    bins = [-1, 12, 36, 60, 72]
    labels = ["New (0-12mo)", "Established (13-36mo)", "Loyal (37-60mo)", "Veteran (60mo+)"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=[b / 72 for b in bins], labels=labels)
    tenure_dist = df["tenure_group"].value_counts()
    logger.info(f"EDA: tenure binning distribution:\n{tenure_dist.to_string()}")

    # ---- 4. Encoding ----
    le = LabelEncoder()
    for col in ["gender", "Partner", "Dependents", "Churn"]:
        df[col + "_enc"] = le.fit_transform(df[col])

    df_encoded = pd.get_dummies(
        df, columns=["Contract", "PaymentMethod", "InternetService"], drop_first=True
    )
    logger.info(
        f"EDA: encoding complete | label-encoded=[gender,Partner,Dependents,Churn] | "
        f"one-hot=[Contract,PaymentMethod,InternetService]"
    )

    # ---- 5. Feature importance (Random Forest) ----
    y = df_encoded["Churn_enc"]
    drop_cols = ["Churn", "Churn_enc", "customerID", "tenure_group", "gender", "Partner", "Dependents"]
    X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number, bool]).astype(float)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    logger.info(f"EDA: top 10 feature importances:\n{importances.head(10).round(4).to_string()}")

    plt.figure(figsize=(6, 5))
    importances.head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/feature_importance.png")
    plt.close()

    # ---- 6. Visualizations: univariate + bivariate ----
    plt.figure(figsize=(4, 4))
    sns.countplot(x="Churn", data=df)
    plt.title("Univariate: Churn Distribution")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/univariate_churn_count.png")
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.histplot(df["MonthlyCharges"], kde=True, bins=30)
    plt.title("Univariate: MonthlyCharges Distribution")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/univariate_monthlycharges.png")
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
    plt.title("Bivariate: MonthlyCharges by Churn")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/bivariate_charges_vs_churn.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    churn_by_contract = pd.crosstab(df["Contract"], df["Churn"], normalize="index")
    churn_by_contract.plot(kind="bar", stacked=True)
    plt.title("Bivariate: Churn Rate by Contract Type")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/bivariate_churn_by_contract.png")
    plt.close()

    logger.info(f"EDA: 5 charts saved to {ARTIFACT_DIR}/")

    df_encoded.to_csv(EDA_OUTPUT_PATH, index=False)
    logger.info(f"EDA: encoded dataset saved to {EDA_OUTPUT_PATH} | shape={df_encoded.shape}")

    return df_encoded


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from ingestion import ingest_data
    from preprocessing import preprocess

    raw_df = ingest_data()
    processed_df, _ = preprocess(raw_df)
    eda_df = run_eda(processed_df)
    print("\nEDA complete. Encoded dataset shape:", eda_df.shape)
    print("Charts saved in artifacts/ folder.")
