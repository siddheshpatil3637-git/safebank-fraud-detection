"""
eda.py
---------------------------------
Requirement 2: Exploratory Data Analysis

Generates and saves (to reports/figures/) the following insights:
    1. Transaction amount distribution
    2. Fraud vs non-fraud comparison
    3. Customer transaction behavior
    4. Transaction frequency analysis
    5. Geographic transaction analysis
    6. Time-based transaction trends
    7. Correlation analysis

Run:
    python src/eda.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import FEATURED_DATA_PATH, FIGURES_DIR, ensure_dirs

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {path}")


def transaction_amount_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df["Transaction_Amount"], bins=60, kde=True, ax=axes[0], color="#2563eb")
    axes[0].set_title("Transaction Amount Distribution")
    axes[0].set_xlabel("Amount")

    sns.boxplot(x=df["Transaction_Amount"], ax=axes[1], color="#2563eb")
    axes[1].set_title("Transaction Amount - Boxplot (Outliers)")
    fig.tight_layout()
    save_fig(fig, "01_transaction_amount_distribution.png")


def fraud_vs_nonfraud(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    counts = df["Fraud_Label"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
    axes[0].pie(counts, labels=counts.index, autopct="%1.2f%%", colors=["#2563eb", "#dc2626"],
                startangle=90, explode=(0, 0.15))
    axes[0].set_title("Fraud vs Legitimate Transactions")

    sns.boxplot(data=df, x="Fraud_Label", y="Transaction_Amount", ax=axes[1], palette=["#2563eb", "#dc2626"])
    axes[1].set_xticklabels(["Legitimate", "Fraud"])
    axes[1].set_title("Transaction Amount: Fraud vs Legitimate")
    axes[1].set_ylim(0, df["Transaction_Amount"].quantile(0.99))
    fig.tight_layout()
    save_fig(fig, "02_fraud_vs_nonfraud.png")


def customer_behavior(df):
    top_customers = df.groupby("Customer_ID")["Transaction_Amount"].sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_customers.plot(kind="barh", ax=ax, color="#0f766e")
    ax.invert_yaxis()
    ax.set_title("Top 15 Customers by Total Transaction Volume")
    ax.set_xlabel("Total Transaction Amount")
    fig.tight_layout()
    save_fig(fig, "03_top_customers_by_volume.png")

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    txn_per_customer = df.groupby("Customer_ID").size()
    sns.histplot(txn_per_customer, bins=40, ax=ax2, color="#7c3aed")
    ax2.set_title("Distribution of Transactions per Customer")
    ax2.set_xlabel("Number of Transactions")
    fig2.tight_layout()
    save_fig(fig2, "04_transactions_per_customer.png")


def frequency_analysis(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    type_counts = df["Transaction_Type"].value_counts()
    sns.barplot(x=type_counts.values, y=type_counts.index, ax=ax, palette="Blues_r")
    ax.set_title("Transaction Frequency by Type")
    ax.set_xlabel("Count")
    fig.tight_layout()
    save_fig(fig, "05_transaction_type_frequency.png")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    cat_counts = df["Merchant_Category"].value_counts()
    sns.barplot(x=cat_counts.values, y=cat_counts.index, ax=ax2, palette="Purples_r")
    ax2.set_title("Transaction Frequency by Merchant Category")
    ax2.set_xlabel("Count")
    fig2.tight_layout()
    save_fig(fig2, "06_merchant_category_frequency.png")


def geographic_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    loc_counts = df["Location"].value_counts()
    sns.barplot(x=loc_counts.values, y=loc_counts.index, ax=axes[0], palette="viridis")
    axes[0].set_title("Transaction Count by Location")
    axes[0].set_xlabel("Count")

    fraud_by_loc = df.groupby("Location")["Fraud_Label"].mean().sort_values(ascending=False) * 100
    sns.barplot(x=fraud_by_loc.values, y=fraud_by_loc.index, ax=axes[1], palette="Reds_r")
    axes[1].set_title("Fraud Rate (%) by Location")
    axes[1].set_xlabel("Fraud Rate (%)")
    fig.tight_layout()
    save_fig(fig, "07_geographic_analysis.png")


def time_trends(df):
    df = df.copy()
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])
    daily = df.set_index("Transaction_Date").resample("D").agg(
        Total_Txns=("Transaction_ID", "count"),
        Fraud_Txns=("Fraud_Label", "sum"),
    )

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(daily.index, daily["Total_Txns"], label="Total Transactions", color="#2563eb")
    ax2 = ax.twinx()
    ax2.plot(daily.index, daily["Fraud_Txns"], label="Fraud Transactions", color="#dc2626")
    ax.set_title("Daily Transaction Volume vs Fraud Count Over Time")
    ax.set_ylabel("Total Transactions")
    ax2.set_ylabel("Fraud Transactions")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    save_fig(fig, "08_daily_trend.png")

    fig2, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    hourly_fraud = df.groupby(df["Transaction_Date"].dt.hour)["Fraud_Label"].mean() * 100
    sns.lineplot(x=hourly_fraud.index, y=hourly_fraud.values, marker="o", ax=axes[0], color="#dc2626")
    axes[0].set_title("Fraud Rate (%) by Hour of Day")
    axes[0].set_xlabel("Hour")

    dow_fraud = df.groupby(df["Transaction_Date"].dt.dayofweek)["Fraud_Label"].mean() * 100
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sns.barplot(x=dow_labels, y=dow_fraud.values, ax=axes[1], palette="Reds")
    axes[1].set_title("Fraud Rate (%) by Day of Week")
    fig2.tight_layout()
    save_fig(fig2, "09_hourly_weekly_fraud_pattern.png")


def correlation_analysis(df):
    numeric_df = df.select_dtypes(include=[np.number])
    # Keep a manageable, meaningful subset for readability
    cols = [c for c in [
        "Transaction_Amount", "Account_Balance", "Fraud_Label", "Txn_Hour",
        "Is_Night_Txn", "Is_Weekend", "Amount_Deviation_Ratio",
        "Amount_Zscore_vs_Customer", "Amount_to_Balance_Ratio",
        "Minutes_Since_Last_Txn", "Is_Rapid_Repeat_Txn", "Is_High_Risk_Location",
        "Is_Cash_Category", "Customer_Location_Diversity",
    ] if c in numeric_df.columns]

    corr = numeric_df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix - Key Features")
    fig.tight_layout()
    save_fig(fig, "10_correlation_heatmap.png")


def run_eda(path: str = FEATURED_DATA_PATH):
    ensure_dirs()
    df = pd.read_csv(path, parse_dates=["Transaction_Date"])
    print(f"Running EDA on {df.shape[0]:,} rows / {df.shape[1]} columns")

    transaction_amount_distribution(df)
    fraud_vs_nonfraud(df)
    customer_behavior(df)
    frequency_analysis(df)
    geographic_analysis(df)
    time_trends(df)
    correlation_analysis(df)

    print("\nEDA complete. All figures saved to reports/figures/")


if __name__ == "__main__":
    run_eda()
