# Architecture

## Pipeline Flow

```
generate_synthetic_data.py
        │
        ▼
data/raw/banking_transactions.csv
        │
        ▼
src/data_preprocessing.py  ──►  data/processed/transactions_clean.csv
        │                        models/label_encoders.joblib
        │                        models/scaler.joblib
        ▼
src/feature_engineering.py ──►  data/processed/transactions_features.csv
        │
        ├──────────────────────────────┐
        ▼                               ▼
src/train_supervised_models.py   src/train_anomaly_models.py
        │                               │
        ▼                               ▼
models/{logreg,dtree,rf,xgb}.joblib   models/{isolation_forest,
reports/metrics/supervised_*          one_class_svm}.joblib
reports/figures/{cm_*, roc_*}         data/processed/transactions_
                                       with_anomaly_scores.csv
        │                               │
        └──────────────┬────────────────┘
                        ▼
              src/risk_scoring.py
                        │
                        ▼
        data/processed/transactions_scored.csv
        data/processed/customer_risk_profiles.csv
                        │
                        ▼
              dashboard/app.py (Streamlit)
```

## Design Decisions

### Why label-encode instead of one-hot?
Tree-based models (Random Forest, XGBoost, Isolation Forest) handle label-encoded
categoricals natively and split on them effectively. One-hot encoding six
categorical columns with double-digit cardinality (Location, Merchant_Category)
would meaningfully inflate the feature space for limited benefit given the
models used. If a linear-only pipeline were required, one-hot encoding should
be reintroduced for `Logistic Regression` specifically.

### Why cap outliers instead of dropping them?
In fraud detection, the outlier IS frequently the fraud. Dropping large/unusual
transactions would silently remove the exact rows the platform needs to detect.
Instead, outliers are winsorized (capped, not deleted) and flagged via
`Is_<column>_Outlier`, preserving the row while preventing a handful of extreme
values from destabilizing scaled features or tree splits.

### Why blend supervised + unsupervised scores?
- Supervised models are strong on **known** fraud typologies present in
  historical labels, but by definition cannot flag patterns they've never
  seen (e.g. a brand-new fraud tactic).
- Unsupervised anomaly detectors need **no labels** and are more robust to
  novel/emerging patterns, at the cost of a higher false-positive rate.
- The weighted blend (60% supervised / 40% anomaly) in `risk_scoring.py`
  ensures a transaction can still be flagged if only one signal fires,
  which mirrors how production fraud systems typically combine rule
  engines, supervised scores, and anomaly detectors.

### Why sample for LOF / One-Class SVM?
Both algorithms have compute complexity that scales poorly (roughly O(n²))
with dataset size, making them impractical on 100k+ rows in a single machine
/ demo environment. Isolation Forest, which scales near-linearly, is run on
the full dataset; LOF and One-Class SVM run on a stratified 40k sample
(oversampling the rare fraud class) to keep training tractable while still
producing meaningful comparisons.

### Extending to a labelless deployment
If a real deployment has no `Fraud_Label` (common in early-stage banking
platforms before enough confirmed fraud cases accumulate):

1. Skip `train_supervised_models.py` entirely.
2. Run only `train_anomaly_models.py` (Isolation Forest as primary).
3. In `risk_scoring.py`, set `SUPERVISED_WEIGHT = 0` and `ANOMALY_WEIGHT = 1`,
   or add a rules-based score component (e.g. hard thresholds on
   `Amount_to_Balance_Ratio`, `Is_Rapid_Repeat_Txn`) to supplement the
   anomaly score until enough confirmed-fraud labels accumulate to train
   a supervised model.

## Data Flow Contracts

| File | Producer | Key columns added |
|---|---|---|
| `transactions_clean.csv` | `data_preprocessing.py` | `*_Encoded`, `*_Scaled`, `Is_*_Outlier` |
| `transactions_features.csv` | `feature_engineering.py` | `Txn_Hour`, `Amount_Deviation_Ratio`, `Amount_to_Balance_Ratio`, etc. |
| `transactions_with_anomaly_scores.csv` | `train_anomaly_models.py` | `IF_Anomaly_Score`, `IF_Is_Anomaly` |
| `transactions_scored.csv` | `risk_scoring.py` | `Fraud_Probability`, `Risk_Score`, `Risk_Tier` |
| `customer_risk_profiles.csv` | `risk_scoring.py` | `Avg_Risk_Score`, `Max_Risk_Score`, `Customer_Risk_Tier` |

The dashboard (`dashboard/app.py`) reads only the final two files plus the
saved model artifacts in `models/` — it never re-runs the pipeline itself,
keeping page loads fast.
