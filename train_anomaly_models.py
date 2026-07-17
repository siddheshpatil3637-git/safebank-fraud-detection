"""
train_anomaly_models.py
---------------------------------
Requirement 3 (Anomaly Detection)

Trains unsupervised anomaly detectors that do NOT require fraud
labels -- critical for a real banking platform where most transactions
are unlabeled or labels arrive with delay:

    - Isolation Forest
    - Local Outlier Factor (LOF)
    - One-Class SVM

Each model outputs an anomaly score per transaction which feeds the
platform's risk scoring engine (src/risk_scoring.py) and the dashboard.
Even though we HAVE fraud labels in this synthetic dataset, we evaluate
the unsupervised models against them purely to sanity-check quality --
in production these models would run label-free.

Run:
    python src/train_anomaly_models.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA

from src.utils import FEATURED_DATA_PATH, FIGURES_DIR, METRICS_DIR, save_model, ensure_dirs

ANOMALY_FEATURES = [
    "Transaction_Amount", "Account_Balance", "Amount_Deviation_Ratio",
    "Amount_Zscore_vs_Customer", "Amount_to_Balance_Ratio",
    "Minutes_Since_Last_Txn", "Txn_Hour", "Is_Night_Txn",
    "Is_High_Risk_Location", "Is_Cash_Category",
]
TARGET_COL = "Fraud_Label"
CONTAMINATION = 0.02  # assumed proportion of anomalies; tuned slightly above true fraud rate


def load_data(sample_size: int = 40000):
    """
    LOF and One-Class SVM are O(n^2)-ish and don't scale to 100k+ rows
    in a demo environment, so we train on a stratified sample. Isolation
    Forest (which DOES scale) is still run on the full dataset separately.
    """
    df = pd.read_csv(FEATURED_DATA_PATH)
    if len(df) > sample_size:
        fraud = df[df[TARGET_COL] == 1]
        legit = df[df[TARGET_COL] == 0].sample(n=sample_size - len(fraud), random_state=42)
        df_sample = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        df_sample = df
    return df, df_sample


def prep_X(df):
    from sklearn.preprocessing import StandardScaler
    X = df[ANOMALY_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def evaluate_anomaly_model(name, y_true, anomaly_flag, anomaly_score, results):
    """anomaly_flag: 1 = anomaly/fraud predicted, 0 = normal. anomaly_score: higher = more anomalous."""
    metrics = {
        "precision": precision_score(y_true, anomaly_flag, zero_division=0),
        "recall": recall_score(y_true, anomaly_flag, zero_division=0),
        "f1_score": f1_score(y_true, anomaly_flag, zero_division=0),
        "roc_auc": roc_auc_score(y_true, anomaly_score) if len(np.unique(y_true)) > 1 else np.nan,
    }
    results[name] = metrics
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics


def run_isolation_forest(df_full):
    X_scaled, scaler = prep_X(df_full)
    model = IsolationForest(
        n_estimators=200, contamination=CONTAMINATION, random_state=42, n_jobs=-1
    )
    model.fit(X_scaled)
    raw_scores = model.decision_function(X_scaled)  # higher = more normal
    anomaly_score = -raw_scores  # flip so higher = more anomalous
    pred = model.predict(X_scaled)  # -1 = anomaly, 1 = normal
    anomaly_flag = (pred == -1).astype(int)

    save_model(model, "isolation_forest")
    save_model(scaler, "isolation_forest_scaler")

    df_full = df_full.copy()
    df_full["IF_Anomaly_Score"] = anomaly_score
    df_full["IF_Is_Anomaly"] = anomaly_flag
    return df_full, anomaly_flag, anomaly_score


def run_lof(df_sample):
    X_scaled, _ = prep_X(df_sample)
    model = LocalOutlierFactor(n_neighbors=25, contamination=CONTAMINATION, novelty=False, n_jobs=-1)
    pred = model.fit_predict(X_scaled)  # -1 anomaly, 1 normal
    anomaly_score = -model.negative_outlier_factor_
    anomaly_flag = (pred == -1).astype(int)
    return anomaly_flag, anomaly_score


def run_one_class_svm(df_sample):
    X_scaled, _ = prep_X(df_sample)
    model = OneClassSVM(kernel="rbf", nu=CONTAMINATION, gamma="scale")
    model.fit(X_scaled)
    raw_scores = model.decision_function(X_scaled)  # higher = more normal
    anomaly_score = -raw_scores
    pred = model.predict(X_scaled)  # -1 anomaly, 1 normal
    anomaly_flag = (pred == -1).astype(int)
    save_model(model, "one_class_svm")
    return anomaly_flag, anomaly_score


def plot_pca_scatter(df_sample, anomaly_flag, name):
    X_scaled, _ = prep_X(df_sample)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 6))
    normal_mask = anomaly_flag == 0
    ax.scatter(coords[normal_mask, 0], coords[normal_mask, 1], s=8, alpha=0.4,
               color="#2563eb", label="Normal")
    ax.scatter(coords[~normal_mask, 0], coords[~normal_mask, 1], s=18, alpha=0.9,
               color="#dc2626", label="Anomaly")
    ax.set_title(f"Anomaly Detection Scatter Plot (PCA) - {name}")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()
    fig.tight_layout()
    fname = f"anomaly_scatter_{name.replace(' ', '_').lower()}.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname))
    plt.close(fig)
    print(f"Saved scatter plot -> {FIGURES_DIR}/{fname}")


def run_training():
    ensure_dirs()
    df_full, df_sample = load_data()
    results = {}

    print("=" * 60)
    print("ISOLATION FOREST (full dataset)")
    print("=" * 60)
    df_full, if_flag_full, if_score_full = run_isolation_forest(df_full)
    evaluate_anomaly_model("Isolation Forest", df_full[TARGET_COL], if_flag_full, if_score_full, results)

    # Re-run IF on the sample too, so all 3 models are plotted on the same population
    df_sample_ids = set(df_sample["Transaction_ID"])
    sample_mask = df_full["Transaction_ID"].isin(df_sample_ids)
    if_flag_sample = df_full.loc[sample_mask, "IF_Is_Anomaly"].values
    plot_pca_scatter(df_sample, if_flag_sample, "Isolation Forest")

    print("\n" + "=" * 60)
    print("LOCAL OUTLIER FACTOR (sampled)")
    print("=" * 60)
    lof_flag, lof_score = run_lof(df_sample)
    evaluate_anomaly_model("Local Outlier Factor", df_sample[TARGET_COL], lof_flag, lof_score, results)
    plot_pca_scatter(df_sample, lof_flag, "Local Outlier Factor")

    print("\n" + "=" * 60)
    print("ONE-CLASS SVM (sampled)")
    print("=" * 60)
    svm_flag, svm_score = run_one_class_svm(df_sample)
    evaluate_anomaly_model("One-Class SVM", df_sample[TARGET_COL], svm_flag, svm_score, results)
    plot_pca_scatter(df_sample, svm_flag, "One-Class SVM")

    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(METRICS_DIR, "anomaly_model_comparison.csv"))
    with open(os.path.join(METRICS_DIR, "anomaly_model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("ANOMALY MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(results_df.round(4))

    # Persist Isolation Forest scores for the full dataset (used by dashboard / risk scoring)
    out_path = os.path.join(os.path.dirname(FEATURED_DATA_PATH), "transactions_with_anomaly_scores.csv")
    df_full.to_csv(out_path, index=False)
    print(f"\nSaved full dataset with anomaly scores -> {out_path}")

    return results_df


if __name__ == "__main__":
    run_training()
