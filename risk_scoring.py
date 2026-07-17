"""
risk_scoring.py
---------------------------------
Requirement 6: Risk scoring mechanism + Predictive fraud detection

Combines the supervised model's fraud probability with the Isolation
Forest anomaly score into a single 0-100 Risk Score per transaction,
plus a Risk Tier (Low / Medium / High / Critical). This is the score
the dashboard and alert system consume.

Design rationale:
    - Supervised probability captures KNOWN fraud patterns.
    - Anomaly score captures NOVEL/unseen patterns the supervised
      model wasn't trained on (important because fraud tactics evolve).
    - Blending both (weighted average) is a common production pattern
      that hedges against either model's blind spots.

Run:
    python src/risk_scoring.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.utils import FEATURED_DATA_PATH, MODELS_DIR, load_model, ensure_dirs
from src.train_supervised_models import FEATURE_COLS

SUPERVISED_WEIGHT = 0.6
ANOMALY_WEIGHT = 0.4


def minmax_scale(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def assign_risk_tier(score: float) -> str:
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 35:
        return "Medium"
    return "Low"


def compute_risk_scores(anomaly_scored_path: str = None) -> pd.DataFrame:
    ensure_dirs()
    if anomaly_scored_path is None:
        anomaly_scored_path = os.path.join(os.path.dirname(FEATURED_DATA_PATH),
                                            "transactions_with_anomaly_scores.csv")
    df = pd.read_csv(anomaly_scored_path, parse_dates=["Transaction_Date"])

    with open(os.path.join(MODELS_DIR, "best_model_name.txt")) as f:
        best_model_name = f.read().strip()
    model_key_map = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    }
    model = load_model(model_key_map.get(best_model_name, "random_forest"))

    X = df[FEATURE_COLS].fillna(0)
    fraud_proba = model.predict_proba(X)[:, 1]

    df["Fraud_Probability"] = fraud_proba
    df["Fraud_Probability_Scaled"] = minmax_scale(pd.Series(fraud_proba)) * 100
    df["Anomaly_Score_Scaled"] = minmax_scale(df["IF_Anomaly_Score"]) * 100

    df["Risk_Score"] = (
        SUPERVISED_WEIGHT * df["Fraud_Probability_Scaled"] +
        ANOMALY_WEIGHT * df["Anomaly_Score_Scaled"]
    ).round(2)

    df["Risk_Tier"] = df["Risk_Score"].apply(assign_risk_tier)

    # Customer-level risk profile: aggregate of recent transaction risk
    cust_risk = df.groupby("Customer_ID")["Risk_Score"].agg(
        Avg_Risk_Score="mean", Max_Risk_Score="max", High_Risk_Txn_Count=lambda s: (s >= 60).sum()
    ).reset_index()
    cust_risk["Customer_Risk_Tier"] = cust_risk["Avg_Risk_Score"].apply(assign_risk_tier)

    out_path = os.path.join(os.path.dirname(FEATURED_DATA_PATH), "transactions_scored.csv")
    df.to_csv(out_path, index=False)

    cust_out_path = os.path.join(os.path.dirname(FEATURED_DATA_PATH), "customer_risk_profiles.csv")
    cust_risk.to_csv(cust_out_path, index=False)

    print(f"Scored dataset saved -> {out_path}")
    print(f"Customer risk profiles saved -> {cust_out_path}")
    print(f"\nRisk tier distribution:\n{df['Risk_Tier'].value_counts()}")

    return df, cust_risk


if __name__ == "__main__":
    compute_risk_scores()
