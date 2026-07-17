"""
run_pipeline.py
---------------------------------
Runs the full SafeBank pipeline end-to-end in the correct order:

    1. Generate synthetic data (skip if data/raw/banking_transactions.csv exists)
    2. Data preprocessing
    3. Feature engineering
    4. EDA (saves figures)
    5. Supervised model training + evaluation
    6. Anomaly detection model training + evaluation
    7. Consolidated evaluation report
    8. Risk scoring

After this completes, launch the dashboard with:
    streamlit run dashboard/app.py

Usage:
    python run_pipeline.py [--force-regenerate-data]
"""

import argparse
import os
import subprocess
import sys
import time

STEPS = [
    ("Data Preprocessing", ["python3", "src/data_preprocessing.py"]),
    ("Feature Engineering", ["python3", "src/feature_engineering.py"]),
    ("Exploratory Data Analysis", ["python3", "src/eda.py"]),
    ("Supervised Model Training", ["python3", "src/train_supervised_models.py"]),
    ("Anomaly Detection Training", ["python3", "src/train_anomaly_models.py"]),
    ("Model Evaluation Report", ["python3", "src/model_evaluation.py"]),
    ("Risk Scoring", ["python3", "src/risk_scoring.py"]),
]


def run_step(name, cmd):
    print("\n" + "=" * 70)
    print(f"STEP: {name}")
    print("=" * 70)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[FAILED] {name} exited with code {result.returncode} after {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"[OK] {name} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-regenerate-data", action="store_true",
                        help="Regenerate the synthetic dataset even if it already exists")
    parser.add_argument("--n_transactions", type=int, default=120000)
    parser.add_argument("--n_customers", type=int, default=3000)
    args = parser.parse_args()

    raw_path = "data/raw/banking_transactions.csv"
    if args.force_regenerate_data or not os.path.exists(raw_path):
        run_step("Synthetic Data Generation", [
            "python3", "generate_synthetic_data.py",
            "--n_customers", str(args.n_customers),
            "--n_transactions", str(args.n_transactions),
        ])
    else:
        print(f"Using existing dataset at {raw_path} (use --force-regenerate-data to rebuild)")

    for name, cmd in STEPS:
        run_step(name, cmd)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("Next: launch the dashboard with:\n    streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
