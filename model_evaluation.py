"""
model_evaluation.py
---------------------------------
Requirement 4: Model Evaluation (consolidated report)

Combines the supervised and anomaly-detection metrics already computed
by train_supervised_models.py / train_anomaly_models.py into a single
markdown report for stakeholders (reports/metrics/model_evaluation_report.md).

Run this AFTER both training scripts:
    python src/train_supervised_models.py
    python src/train_anomaly_models.py
    python src/model_evaluation.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.utils import METRICS_DIR


def build_report():
    sup_path = os.path.join(METRICS_DIR, "supervised_model_comparison.csv")
    ano_path = os.path.join(METRICS_DIR, "anomaly_model_comparison.csv")

    sup_df = pd.read_csv(sup_path, index_col=0) if os.path.exists(sup_path) else None
    ano_df = pd.read_csv(ano_path, index_col=0) if os.path.exists(ano_path) else None

    lines = ["# SafeBank Model Evaluation Report", ""]

    if sup_df is not None:
        lines.append("## Supervised Fraud Classification Models\n")
        lines.append(sup_df.round(4).to_markdown())
        best = sup_df["roc_auc"].idxmax()
        lines.append(f"\n**Best supervised model (by ROC-AUC): `{best}`**\n")
        lines.append(
            "\nNote: Accuracy is a misleading metric here due to the ~1.4% fraud "
            "prevalence -- a model predicting 'legitimate' for every transaction "
            "would score >98% accuracy while catching zero fraud. Precision, "
            "Recall, F1, and ROC-AUC are the metrics that matter for this task.\n"
        )

    if ano_df is not None:
        lines.append("\n## Unsupervised Anomaly Detection Models\n")
        lines.append(ano_df.round(4).to_markdown())
        best_ano = ano_df["roc_auc"].idxmax()
        lines.append(f"\n**Best anomaly detector (by ROC-AUC): `{best_ano}`**\n")
        lines.append(
            "\nThese models were evaluated against fraud labels purely for "
            "sanity-checking; in production they can flag anomalies WITHOUT "
            "any labeled data, which is their core advantage over the "
            "supervised models above.\n"
        )

    lines.append(
        "\n## Recommended Production Approach\n"
        "- Use the best supervised model for known-pattern fraud scoring.\n"
        "- Use Isolation Forest in parallel to catch novel/emerging fraud patterns "
        "the supervised model hasn't seen.\n"
        "- Blend both into the platform's unified Risk Score (see "
        "`src/risk_scoring.py`) so a transaction can be flagged even if only "
        "one of the two signals fires.\n"
    )

    report_path = os.path.join(METRICS_DIR, "model_evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Evaluation report saved -> {report_path}")


if __name__ == "__main__":
    build_report()
