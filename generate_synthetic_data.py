"""
generate_synthetic_data.py
---------------------------------
Generates a realistic synthetic banking transaction dataset for the
SafeBank Transaction Anomaly Detection Platform.

Why synthetic data?
No real dataset was supplied with this project. Real fraud datasets
(e.g. from Kaggle) are heavily anonymized (PCA components) and lack the
rich categorical fields (Merchant_Category, Device_Type, Location) that
this platform is designed around. This script generates a dataset with
realistic distributions, seasonality, and an injected fraud pattern so
every downstream script (EDA, preprocessing, modeling, dashboard) can be
demonstrated end-to-end.

To use YOUR OWN dataset instead: drop a CSV with the same column names
into data/raw/banking_transactions.csv and skip this script.

Usage:
    python generate_synthetic_data.py --n_customers 3000 --n_transactions 120000
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

MERCHANT_CATEGORIES = [
    "Grocery", "Electronics", "Travel", "Restaurant", "Fuel",
    "Online Shopping", "Utilities", "Healthcare", "Entertainment",
    "Jewelry", "ATM Withdrawal", "Money Transfer"
]

TRANSACTION_TYPES = ["Debit", "Credit", "Transfer", "Withdrawal", "Deposit", "Online Payment"]

DEVICE_TYPES = ["Mobile App", "Web Browser", "ATM", "POS Terminal", "Phone Banking"]

LOCATIONS = [
    "Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Kochi",
    "Singapore", "Dubai", "London", "New York"
]

HIGH_RISK_LOCATIONS = {"Dubai", "London", "New York", "Singapore"}  # cross-border = higher inherent risk


def generate_customers(n_customers: int) -> pd.DataFrame:
    """Create a base customer profile table (home city, avg balance, risk appetite)."""
    customer_ids = [f"CUST{100000 + i}" for i in range(n_customers)]
    home_location = RNG.choice(LOCATIONS, size=n_customers, p=_location_probs())
    base_balance = RNG.lognormal(mean=10.5, sigma=0.9, size=n_customers)  # skewed, realistic
    # A small fraction of customers are "compromised" and generate more fraud
    compromised = RNG.random(n_customers) < 0.03
    return pd.DataFrame({
        "Customer_ID": customer_ids,
        "Home_Location": home_location,
        "Base_Balance": base_balance,
        "Is_Compromised_Profile": compromised,
    })


def _location_probs():
    # weight domestic cities much higher than international ones
    weights = np.array([1 if loc not in HIGH_RISK_LOCATIONS else 0.15 for loc in LOCATIONS], dtype=float)
    return weights / weights.sum()


def generate_transactions(customers: pd.DataFrame, n_transactions: int, start_date: str, end_date: str) -> pd.DataFrame:
    n_customers = len(customers)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    span_seconds = int((end - start).total_seconds())

    cust_idx = RNG.integers(0, n_customers, size=n_transactions)
    cust_rows = customers.iloc[cust_idx].reset_index(drop=True)

    # Transaction timestamps: mildly seasonal (more txns during daytime hours)
    random_seconds = RNG.integers(0, span_seconds, size=n_transactions)
    timestamps = start + pd.to_timedelta(random_seconds, unit="s")
    hour_bias = RNG.normal(loc=14, scale=4, size=n_transactions).clip(0, 23).astype(int)
    timestamps = pd.to_datetime(timestamps.date) + pd.to_timedelta(hour_bias, unit="h") + \
                 pd.to_timedelta(RNG.integers(0, 60, n_transactions), unit="m")

    transaction_type = RNG.choice(TRANSACTION_TYPES, size=n_transactions, p=[0.28, 0.15, 0.17, 0.12, 0.13, 0.15])
    merchant_category = RNG.choice(MERCHANT_CATEGORIES, size=n_transactions)
    device_type = RNG.choice(DEVICE_TYPES, size=n_transactions, p=[0.40, 0.28, 0.12, 0.15, 0.05])

    # Most transactions happen near the customer's home location
    away_from_home = RNG.random(n_transactions) < 0.08
    location = np.where(
        away_from_home,
        RNG.choice(LOCATIONS, size=n_transactions),
        cust_rows["Home_Location"].values,
    )

    # Transaction amount: lognormal, category-dependent scaling
    category_scale = pd.Series(merchant_category).map({
        "Grocery": 1.0, "Electronics": 4.5, "Travel": 6.0, "Restaurant": 0.8,
        "Fuel": 1.2, "Online Shopping": 2.0, "Utilities": 1.5, "Healthcare": 3.0,
        "Entertainment": 1.1, "Jewelry": 8.0, "ATM Withdrawal": 2.5, "Money Transfer": 5.0,
    }).values
    base_amount = RNG.lognormal(mean=6.5, sigma=1.0, size=n_transactions)
    transaction_amount = np.round(base_amount * category_scale, 2)

    account_balance = np.round(
        np.maximum(cust_rows["Base_Balance"].values + RNG.normal(0, 5000, n_transactions), 0), 2
    )

    transaction_status = RNG.choice(
        ["Success", "Failed", "Pending"], size=n_transactions, p=[0.93, 0.05, 0.02]
    )

    df = pd.DataFrame({
        "Transaction_ID": [f"TXN{500000 + i}" for i in range(n_transactions)],
        "Customer_ID": cust_rows["Customer_ID"].values,
        "Transaction_Amount": transaction_amount,
        "Transaction_Type": transaction_type,
        "Transaction_Date": timestamps,
        "Merchant_Category": merchant_category,
        "Location": location,
        "Device_Type": device_type,
        "Account_Balance": account_balance,
        "Transaction_Status": transaction_status,
    })

    df["_is_compromised"] = cust_rows["Is_Compromised_Profile"].values
    df["_away_from_home"] = away_from_home
    return df


def inject_fraud_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject a realistic fraud pattern (~1.2% fraud rate) driven by a
    combination of risk factors, rather than pure randomness, so
    supervised models have real signal to learn.
    """
    n = len(df)
    risk_score = np.zeros(n)

    # Large amount relative to account balance
    ratio = df["Transaction_Amount"] / (df["Account_Balance"] + 1)
    risk_score += (ratio > 0.5).astype(int) * 2.5

    # High absolute amount
    high_amount_thresh = df["Transaction_Amount"].quantile(0.97)
    risk_score += (df["Transaction_Amount"] > high_amount_thresh).astype(int) * 2.0

    # Odd hours (midnight - 5am)
    risk_score += df["Transaction_Date"].dt.hour.isin(range(0, 5)).astype(int) * 1.5

    # Cross border / away from home location
    risk_score += df["_away_from_home"].astype(int) * 1.5
    risk_score += df["Location"].isin(HIGH_RISK_LOCATIONS).astype(int) * 1.0

    # Compromised customer profile
    risk_score += df["_is_compromised"].astype(int) * 3.0

    # High-risk categories for laundering / cash-out
    risk_score += df["Merchant_Category"].isin(["Money Transfer", "ATM Withdrawal", "Jewelry"]).astype(int) * 1.0

    # Online/web device slightly higher risk than in-person POS
    risk_score += df["Device_Type"].isin(["Web Browser", "Mobile App"]).astype(int) * 0.5

    # Convert score to probability via logistic squashing, then sample
    prob = 1 / (1 + np.exp(-(risk_score - 6.5)))
    # Add small base noise so not 100% deterministic
    prob = np.clip(prob + RNG.normal(0, 0.03, n), 0, 1)
    fraud_label = (RNG.random(n) < prob).astype(int)

    # Ensure a reasonable fraud rate (~1-2%) by rebalancing if needed
    target_rate = 0.014
    current_rate = fraud_label.mean()
    if current_rate > 0:
        # Downsample fraud flags proportionally if too high
        if current_rate > target_rate * 1.5:
            keep_prob = target_rate / current_rate
            drop_mask = (fraud_label == 1) & (RNG.random(n) > keep_prob)
            fraud_label[drop_mask] = 0

    df["Fraud_Label"] = fraud_label
    df = df.drop(columns=["_is_compromised", "_away_from_home"])
    return df


def main(n_customers: int, n_transactions: int, start_date: str, end_date: str, out_path: str):
    print(f"Generating {n_customers} customers and {n_transactions} transactions...")
    customers = generate_customers(n_customers)
    df = generate_transactions(customers, n_transactions, start_date, end_date)
    df = inject_fraud_labels(df)
    df = df.sort_values("Transaction_Date").reset_index(drop=True)

    # Introduce a small amount of realistic messiness (missing values, dupes)
    # so the preprocessing pipeline has something meaningful to clean.
    missing_idx = RNG.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[missing_idx, "Account_Balance"] = np.nan
    missing_idx2 = RNG.choice(df.index, size=int(0.005 * len(df)), replace=False)
    df.loc[missing_idx2, "Merchant_Category"] = np.nan

    dupe_rows = df.sample(n=int(0.003 * len(df)), random_state=1)
    df = pd.concat([df, dupe_rows], ignore_index=True)

    df.to_csv(out_path, index=False)
    fraud_rate = df["Fraud_Label"].mean() * 100
    print(f"Saved dataset to {out_path}")
    print(f"Total rows: {len(df):,} | Fraud rate: {fraud_rate:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic SafeBank transaction dataset")
    parser.add_argument("--n_customers", type=int, default=3000)
    parser.add_argument("--n_transactions", type=int, default=120000)
    parser.add_argument("--start_date", type=str, default="2024-07-01")
    parser.add_argument("--end_date", type=str, default="2025-07-01")
    parser.add_argument("--out_path", type=str, default="data/raw/banking_transactions.csv")
    args = parser.parse_args()
    main(args.n_customers, args.n_transactions, args.start_date, args.end_date, args.out_path)
