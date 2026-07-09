"""
generate_sample_data.py
------------------------
Generates a synthetic dataset with the SAME column structure as the real
Kaggle 'Telco Customer Churn' dataset (https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

WHY THIS EXISTS:
If you haven't downloaded the real Kaggle CSV yet, run this script to instantly
get a working data/telco_churn_raw.csv so you can test the ENTIRE pipeline end to end.

TO USE THE REAL DATASET INSTEAD:
1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download WA_Fn-UseC_-Telco-Customer-Churn.csv
3. Rename it to telco_churn_raw.csv and place it inside the data/ folder
4. Skip running this script
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000  # number of synthetic customers (Kaggle's real file has 7043)


def generate():
    customer_id = [f"C{10000+i}" for i in range(N)]
    gender = np.random.choice(["Male", "Female"], N)
    senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], N)
    dependents = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])
    tenure = np.random.randint(0, 73, N)

    phone_service = np.random.choice(["Yes", "No"], N, p=[0.9, 0.1])
    multiple_lines = np.random.choice(["Yes", "No", "No phone service"], N, p=[0.42, 0.48, 0.1])
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])
    online_security = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.29, 0.49, 0.22])
    online_backup = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.34, 0.44, 0.22])
    device_protection = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.34, 0.44, 0.22])
    tech_support = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.29, 0.49, 0.22])
    streaming_tv = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.38, 0.40, 0.22])
    streaming_movies = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.38, 0.40, 0.22])

    contract = np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        N, p=[0.34, 0.23, 0.22, 0.21],
    )

    monthly_charges = np.round(np.random.uniform(18, 120, N), 2)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 50, N), 2)
    total_charges = np.clip(total_charges, 0, None)

    # Inject some missing values in TotalCharges (mirrors the real dataset's quirk
    # where brand-new customers, tenure=0, have blank TotalCharges)
    total_charges_str = total_charges.astype(str)
    blank_idx = np.where(tenure == 0)[0]
    for i in blank_idx:
        total_charges_str[i] = " "

    # Build churn probability from a logical combination of features (so the
    # dataset has REAL learnable signal, not pure noise)
    churn_score = (
        (contract == "Month-to-month") * 0.35
        + (internet_service == "Fiber optic") * 0.15
        + (payment_method == "Electronic check") * 0.15
        + (tenure < 12) * 0.25
        + (monthly_charges > 80) * 0.10
        - (tech_support == "Yes") * 0.15
        - (online_security == "Yes") * 0.10
        + np.random.normal(0, 0.15, N)
    )
    churn_prob = 1 / (1 + np.exp(-5 * (churn_score - 0.35)))
    churn = np.where(np.random.rand(N) < churn_prob, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges_str,
        "Churn": churn,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/telco_churn_raw.csv", index=False)
    print(f"Synthetic dataset created: data/telco_churn_raw.csv  (shape={df.shape})")
    print(df["Churn"].value_counts(normalize=True).round(3))
