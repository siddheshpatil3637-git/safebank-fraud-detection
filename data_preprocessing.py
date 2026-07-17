"""
data_preprocessing.py
---------------------------------
Requirement 1: Data Understanding and Preprocessing

Steps performed:
    1. Load and inspect the dataset
    2. Handle missing values
    3. Remove duplicate records
    4. Detect and treat outliers (IQR capping on amount/balance)
    5. Encode categorical variables (Label/One-Hot depending on cardinality)
    6. Scale numerical variables (StandardScaler)
    7. Basic feature engineering (see feature_engineering.py for the
       richer, model-facing feature set)

Run:
    python src/data_preprocessing.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, CATEGORICAL_COLS, NUMERIC_COLS,
    ensure_dirs,
)


def load_and_inspect(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Transaction_Date"])
    print("=" * 60)
    print("DATA INSPECTION")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDuplicate rows: {df.duplicated().shape[0] - df.drop_duplicates().shape[0]}")
    if "Fraud_Label" in df.columns:
        print(f"\nFraud rate: {df['Fraud_Label'].mean() * 100:.2f}%")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
    - Numeric columns: median imputation per Transaction_Type (more
      representative than a single global median for a skewed field
      like account balance).
    - Categorical columns: mode imputation ("Unknown" fallback).
    """
    df = df.copy()

    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            df[col] = df.groupby("Transaction_Type")[col].transform(
                lambda s: s.fillna(s.median())
            )
            df[col] = df[col].fillna(df[col].median())  # safety net

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["Transaction_ID"]).reset_index(drop=True)
    # Also guard against fully-duplicated rows with different generated IDs
    df = df.drop_duplicates(
        subset=[c for c in df.columns if c != "Transaction_ID"]
    ).reset_index(drop=True)
    after = len(df)
    print(f"Removed {before - after} duplicate records ({before} -> {after})")
    return df


def treat_outliers(df: pd.DataFrame, cols=NUMERIC_COLS, factor: float = 3.0) -> pd.DataFrame:
    """
    Cap (winsorize) extreme outliers using the IQR method rather than
    dropping them outright -- in fraud detection, outliers are often
    the *signal*, not noise, so we flag them instead of deleting rows.
    A wide factor (3x IQR) is used to avoid capping legitimate large
    transactions, while an `Is_Amount_Outlier` flag preserves the
    information for modeling.
    """
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        flag_col = f"Is_{col}_Outlier"
        df[flag_col] = ((df[col] < lower) | (df[col] > upper)).astype(int)
        df[col] = df[col].clip(lower=max(lower, 0), upper=upper)
    return df


def encode_categoricals(df: pd.DataFrame, cols=CATEGORICAL_COLS):
    """
    Label-encode categoricals and keep the fitted encoders so the
    dashboard / inference pipeline can apply the exact same mapping.
    One-hot is avoided here because tree-based models (RF/XGBoost)
    handle label-encoded categoricals well and it keeps the feature
    space compact for a platform with several categorical fields.
    """
    df = df.copy()
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        df[f"{col}_Encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def scale_numeric(df: pd.DataFrame, cols=NUMERIC_COLS):
    df = df.copy()
    scaler = StandardScaler()
    scaled_cols = [f"{c}_Scaled" for c in cols]
    df[scaled_cols] = scaler.fit_transform(df[cols])
    return df, scaler


def run_pipeline(raw_path: str = RAW_DATA_PATH, out_path: str = PROCESSED_DATA_PATH):
    ensure_dirs()
    df = load_and_inspect(raw_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = treat_outliers(df)
    df, encoders = encode_categoricals(df)
    df, scaler = scale_numeric(df)

    import joblib
    from src.utils import MODELS_DIR
    joblib.dump(encoders, os.path.join(MODELS_DIR, "label_encoders.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    df.to_csv(out_path, index=False)
    print(f"\nProcessed dataset saved -> {out_path}")
    print(f"Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    run_pipeline()
