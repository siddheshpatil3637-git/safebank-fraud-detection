# SafeBank Model Evaluation Report

## Supervised Fraud Classification Models

|                     |   accuracy |   precision |   recall |   f1_score |   roc_auc |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Logistic Regression |     0.7447 |      0.0298 |   0.5439 |     0.0564 |    0.6952 |
| Decision Tree       |     0.9141 |      0.0702 |   0.4181 |     0.1202 |    0.6677 |
| Random Forest       |     0.966  |      0.1324 |   0.2565 |     0.1746 |    0.6363 |
| XGBoost             |     0.9529 |      0.0853 |   0.2423 |     0.1262 |    0.6449 |

**Best supervised model (by ROC-AUC): `Logistic Regression`**


Note: Accuracy is a misleading metric here due to the ~1.4% fraud prevalence -- a model predicting 'legitimate' for every transaction would score >98% accuracy while catching zero fraud. Precision, Recall, F1, and ROC-AUC are the metrics that matter for this task.


## Unsupervised Anomaly Detection Models

|                      |   precision |   recall |   f1_score |   roc_auc |
|:---------------------|------------:|---------:|-----------:|----------:|
| Isolation Forest     |      0.0729 |   0.104  |     0.0857 |    0.6744 |
| Local Outlier Factor |      0.07   |   0.0333 |     0.0451 |    0.5252 |
| One-Class SVM        |      0.1004 |   0.0481 |     0.0651 |    0.5649 |

**Best anomaly detector (by ROC-AUC): `Isolation Forest`**


These models were evaluated against fraud labels purely for sanity-checking; in production they can flag anomalies WITHOUT any labeled data, which is their core advantage over the supervised models above.


## Recommended Production Approach
- Use the best supervised model for known-pattern fraud scoring.
- Use Isolation Forest in parallel to catch novel/emerging fraud patterns the supervised model hasn't seen.
- Blend both into the platform's unified Risk Score (see `src/risk_scoring.py`) so a transaction can be flagged even if only one of the two signals fires.
