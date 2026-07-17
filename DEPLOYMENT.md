# Deployment Guide

## Option A — Streamlit Community Cloud (fastest, free tier)

1. Push this project to a GitHub repository.
2. Ensure `data/processed/transactions_scored.csv`, `data/processed/customer_risk_profiles.csv`
   and the `models/` directory are committed (or run `run_pipeline.py` in a
   build step — see `packages.txt`/`requirements.txt` note below), since the
   dashboard reads these directly rather than regenerating them on every load.
3. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub
   account, select the repo, and set:
   - **Main file path**: `dashboard/app.py`
   - **Python version**: 3.10+
4. Streamlit Cloud installs `requirements.txt` automatically. Deploy.

> If the repo is public and you don't want to commit generated data/models,
> add a one-time setup step in a `streamlit_app.py` wrapper that calls
> `run_pipeline.py` on first boot if `data/processed/transactions_scored.csv`
> is missing — accept the slower cold start in exchange for a lean repo.

## Option B — Docker (self-hosted / any cloud VM, ECS, Cloud Run)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the pipeline once at build time so the image ships with scored data
RUN python run_pipeline.py

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t safebank-dashboard .
docker run -p 8501:8501 safebank-dashboard
```

Push the image to any container registry (ECR, GCR, Docker Hub) and deploy to
ECS Fargate, Cloud Run, Azure Container Apps, or a plain VM behind nginx with
TLS termination.

## Option C — Flask scoring API (headless, for system-to-system integration)

If another banking system needs to call the risk engine programmatically
rather than through the dashboard, wrap the scoring logic in a minimal Flask
API:

```python
# api/app.py
from flask import Flask, request, jsonify
import pandas as pd
from src.utils import load_model
from src.train_supervised_models import FEATURE_COLS

app = Flask(__name__)
clf = load_model("random_forest")          # or best_model_name.txt
iso_forest = load_model("isolation_forest")
iso_scaler = load_model("isolation_forest_scaler")

@app.route("/score", methods=["POST"])
def score_transaction():
    payload = request.get_json()
    row = pd.DataFrame([payload])
    X = row[FEATURE_COLS].fillna(0)
    fraud_proba = float(clf.predict_proba(X)[0, 1])
    return jsonify({"fraud_probability": fraud_proba})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

Run with `python api/app.py` or behind gunicorn (`gunicorn -w 4 -b 0.0.0.0:5000 api.app:app`)
for production traffic.

## Production Hardening Checklist (banking context)

Before any real deployment handling actual customer data:

- [ ] Authentication + role-based access control on the dashboard (Streamlit
      supports this via `streamlit-authenticator` or an upstream reverse
      proxy with SSO).
- [ ] TLS everywhere (dashboard, any API endpoints).
- [ ] Encrypt data at rest (processed CSVs / model artifacts) and in transit.
- [ ] PII masking/tokenization for `Customer_ID`, `Location` in any
      non-production environment or logs.
- [ ] Audit logging of every alert view/dismissal for regulatory traceability.
- [ ] Move from CSV batch scoring to a streaming pipeline (Kafka/Kinesis)
      for true real-time detection at transaction-posting time.
- [ ] Model governance: versioning, approval workflow, and periodic
      fairness/bias review before each model promotion to production.
