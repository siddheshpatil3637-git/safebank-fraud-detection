"""
utils.py
---------------------------------
Shared helper functions used across preprocessing, EDA, modeling and
the dashboard. Centralizing these avoids duplicated logic and keeps
column names / paths consistent across the project.
"""

import os
import joblib
import pandas as pd

# ---------------------------------------------------------------------
# Paths (relative to project root; scripts assume they're run from root
# or that PROJECT_ROOT is adjusted accordingly)
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "banking_transactions.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "transactions_clean.csv")
FEATURED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "transactions_features.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
METRICS_DIR = os.path.join(PROJECT_ROOT, "reports", "metrics")

CATEGORICAL_COLS = ["Transaction_Type", "Merchant_Category", "Location", "Device_Type", "Transaction_Status"]
NUMERIC_COLS = ["Transaction_Amount", "Account_Balance"]
TARGET_COL = "Fraud_Label"


def ensure_dirs():
    for d in [os.path.dirname(PROCESSED_DATA_PATH), MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
        os.makedirs(d, exist_ok=True)


def save_model(model, name: str):
    ensure_dirs()
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"Model saved -> {path}")
    return path


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    return joblib.load(path)


def load_csv(path: str, parse_dates=None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=parse_dates)
