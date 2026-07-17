"""
feature_engineering.py
---------------------------------
Builds the model-facing feature set on top of the cleaned data. These
features encode customer behavior patterns that are highly predictive
of fraud in real banking systems:

    - Time-based features (hour, day-of-week, is_night, is_weekend)
    - Customer aggregate behavior (rolling avg spend, txn frequency)
    - Deviation features (how unusual is THIS txn vs the customer's norm)
    - Balance-to-amount ratio
    - Velocity features (transactions in short time windows)

Run:
    python src/feature_engineering.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.utils import PROCESSED_DATA_PATH, FEATURED_DATA_PATH, ensure_dirs


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])
    df["Txn_Hour"] = df["Transaction_Date"].dt.hour
    df["Txn_DayOfWeek"] = df["Transaction_Date"].dt.dayofweek  # 0=Mon
    df["Txn_Day"] = df["Transaction_Date"].dt.day
    df["Txn_Month"] = df["Transaction_Date"].dt.month
    df["Is_Night_Txn"] = df["Txn_Hour"].isin(range(0, 5)).astype(int)
    df["Is_Weekend"] = (df["Txn_DayOfWeek"] >= 5).astype(int)
    return df


def add_customer_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Customer-level aggregates that describe 'normal' behavior, then
    per-transaction deviation from that norm -- the core signal used
    by real-world fraud systems (e.g. 'this transaction is 8x this
    customer's average spend').
    """
    df = df.copy()
    df = df.sort_values(["Customer_ID", "Transaction_Date"]).reset_index(drop=True)

    cust_stats = df.groupby("Customer_ID")["Transaction_Amount"].agg(
        Customer_Avg_Amount="mean",
        Customer_Std_Amount="std",
        Customer_Txn_Count="count",
    ).reset_index()
    cust_stats["Customer_Std_Amount"] = cust_stats["Customer_Std_Amount"].fillna(0)

    df = df.merge(cust_stats, on="Customer_ID", how="left")

    # Deviation of this transaction from customer's own historical average
    df["Amount_Deviation_Ratio"] = df["Transaction_Amount"] / (df["Customer_Avg_Amount"] + 1)
    df["Amount_Zscore_vs_Customer"] = (
        (df["Transaction_Amount"] - df["Customer_Avg_Amount"]) /
        (df["Customer_Std_Amount"] + 1)
    )

    # Balance-to-amount ratio: how much of the account this txn represents
    df["Amount_to_Balance_Ratio"] = df["Transaction_Amount"] / (df["Account_Balance"] + 1)

    # Time since previous transaction for the same customer (velocity signal)
    df["Prev_Txn_Time"] = df.groupby("Customer_ID")["Transaction_Date"].shift(1)
    df["Minutes_Since_Last_Txn"] = (
        (df["Transaction_Date"] - df["Prev_Txn_Time"]).dt.total_seconds() / 60
    )
    df["Minutes_Since_Last_Txn"] = df["Minutes_Since_Last_Txn"].fillna(
        df["Minutes_Since_Last_Txn"].median()
    )
    df["Is_Rapid_Repeat_Txn"] = (df["Minutes_Since_Last_Txn"] < 5).astype(int)

    # Rolling transaction count in the customer's history so far (cumulative)
    df["Customer_Cumulative_Txns"] = df.groupby("Customer_ID").cumcount() + 1

    df = df.drop(columns=["Prev_Txn_Time"])
    return df


def add_location_device_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    HIGH_RISK_LOCATIONS = {"Dubai", "London", "New York", "Singapore"}
    df["Is_High_Risk_Location"] = df["Location"].isin(HIGH_RISK_LOCATIONS).astype(int)
    df["Is_Cash_Category"] = df["Merchant_Category"].isin(
        ["ATM Withdrawal", "Money Transfer", "Jewelry"]
    ).astype(int)

    # Has the customer used more than one location historically? (location switching)
    loc_counts = df.groupby("Customer_ID")["Location"].transform("nunique")
    df["Customer_Location_Diversity"] = loc_counts
    return df


def run_pipeline(in_path: str = PROCESSED_DATA_PATH, out_path: str = FEATURED_DATA_PATH):
    ensure_dirs()
    df = pd.read_csv(in_path, parse_dates=["Transaction_Date"])
    df = add_time_features(df)
    df = add_customer_behavior_features(df)
    df = add_location_device_risk_features(df)

    df.to_csv(out_path, index=False)
    print(f"Feature-engineered dataset saved -> {out_path}")
    print(f"Shape: {df.shape}")
    print(f"New feature columns added: "
          f"{[c for c in df.columns if c not in pd.read_csv(in_path, nrows=1).columns]}")
    return df


if __name__ == "__main__":
    run_pipeline()
