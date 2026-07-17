"""
train_supervised_models.py
---------------------------------
Requirement 3 (Supervised) + Requirement 4 (Evaluation)

Trains and evaluates:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost

Handles severe class imbalance (~1.4% fraud) via class_weight /
scale_pos_weight rather than naive oversampling, and evaluates with
metrics appropriate for imbalanced fraud detection (Precision, Recall,
F1, ROC-AUC -- NOT plain accuracy, which is misleading here).

Run:
    python src/train_supervised_models.py
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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from xgboost import XGBClassifier

from src.utils import FEATURED_DATA_PATH, MODELS_DIR, FIGURES_DIR, METRICS_DIR, save_model, ensure_dirs

FEATURE_COLS = [
    "Transaction_Amount", "Account_Balance",
    "Transaction_Type_Encoded", "Merchant_Category_Encoded", "Location_Encoded",
    "Device_Type_Encoded", "Transaction_Status_Encoded",
    "Txn_Hour", "Txn_DayOfWeek", "Is_Night_Txn", "Is_Weekend",
    "Amount_Deviation_Ratio", "Amount_Zscore_vs_Customer", "Amount_to_Balance_Ratio",
    "Minutes_Since_Last_Txn", "Is_Rapid_Repeat_Txn", "Customer_Cumulative_Txns",
    "Is_High_Risk_Location", "Is_Cash_Category", "Customer_Location_Diversity",
    "Is_Transaction_Amount_Outlier", "Is_Account_Balance_Outlier",
]
TARGET_COL = "Fraud_Label"


def load_data():
    """
    feature_engineering.py builds on top of the already-encoded/outlier-
    flagged columns produced by data_preprocessing.py, so the featured
    file alone already contains everything the models need.
    """
    df = pd.read_csv(FEATURED_DATA_PATH)
    return df


def get_splits(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Train fraud rate: {y_train.mean()*100:.2f}% | Test fraud rate: {y_test.mean()*100:.2f}%")
    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test, results: dict):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    results[name] = metrics

    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Legitimate", "Fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"cm_{name.replace(' ', '_').lower()}.png"))
    plt.close(fig)

    return y_proba


def plot_roc_curves(y_test, proba_dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Supervised Fraud Models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "roc_curves_supervised.png"))
    plt.close(fig)
    print(f"Saved ROC comparison -> {FIGURES_DIR}/roc_curves_supervised.png")


def run_training():
    ensure_dirs()
    df = load_data()
    X_train, X_test, y_train, y_test = get_splits(df)

    results = {}
    proba_dict = {}

    # 1. Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    log_reg.fit(X_train, y_train)
    proba_dict["Logistic Regression"] = evaluate_model("Logistic Regression", log_reg, X_test, y_test, results)
    save_model(log_reg, "logistic_regression")

    # 2. Decision Tree
    dtree = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42)
    dtree.fit(X_train, y_train)
    proba_dict["Decision Tree"] = evaluate_model("Decision Tree", dtree, X_test, y_test, results)
    save_model(dtree, "decision_tree")

    # 3. Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced",
        n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    proba_dict["Random Forest"] = evaluate_model("Random Forest", rf, X_test, y_test, results)
    save_model(rf, "random_forest")

    # 4. XGBoost
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    proba_dict["XGBoost"] = evaluate_model("XGBoost", xgb, X_test, y_test, results)
    save_model(xgb, "xgboost")

    plot_roc_curves(y_test, proba_dict)

    # Feature importance (from the best tree-based model: XGBoost)
    importances = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    importances.head(15).plot(kind="barh", ax=ax, color="#0f766e")
    ax.invert_yaxis()
    ax.set_title("Top 15 Feature Importances (XGBoost)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_importance_xgboost.png"))
    plt.close(fig)

    # Save comparison metrics
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(METRICS_DIR, "supervised_model_comparison.csv"))
    with open(os.path.join(METRICS_DIR, "supervised_model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(results_df.round(4))

    best_model_name = results_df["roc_auc"].idxmax()
    print(f"\nBest model by ROC-AUC: {best_model_name}")

    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_model_name)

    return results_df


if __name__ == "__main__":
    run_training()
