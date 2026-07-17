"""
SafeBank Transaction Anomaly Detection Platform - Interactive Dashboard
---------------------------------------------------------------------
Run:
    streamlit run dashboard/app.py

Sections:
    1. Transaction Monitoring  - KPIs (total txns, suspicious count, high-risk customers, fraud %)
    2. Anomaly Visualization   - risk score distribution, trend charts, PCA scatter, heatmaps, geo map
    3. Risk Analysis           - fraud probability indicators, customer risk profiles, alerts
    4. Real-Time Simulation    - streams synthetic transactions and scores them live
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.utils import PROJECT_ROOT, MODELS_DIR, load_model
from src.train_supervised_models import FEATURE_COLS
from src.risk_scoring import assign_risk_tier, minmax_scale

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "transactions_scored.csv")
CUST_RISK_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "customer_risk_profiles.csv")

st.set_page_config(
    page_title="SafeBank | Transaction Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown("""
<style>
.kpi-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 14px;
    padding: 18px 20px;
    color: white;
    border: 1px solid #334155;
}
.kpi-value { font-size: 28px; font-weight: 700; margin: 4px 0; }
.kpi-label { font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
.risk-critical { color: #ef4444; font-weight: 700; }
.risk-high { color: #f97316; font-weight: 700; }
.risk-medium { color: #eab308; font-weight: 700; }
.risk-low { color: #22c55e; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading transaction data...")
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Transaction_Date"])
    cust_risk = pd.read_csv(CUST_RISK_PATH)
    return df, cust_risk


@st.cache_resource
def load_scoring_models():
    with open(os.path.join(MODELS_DIR, "best_model_name.txt")) as f:
        best_name = f.read().strip()
    key_map = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
               "Random Forest": "random_forest", "XGBoost": "xgboost"}
    clf = load_model(key_map.get(best_name, "random_forest"))
    iso_forest = load_model("isolation_forest")
    iso_scaler = load_model("isolation_forest_scaler")
    return clf, iso_forest, iso_scaler, best_name


def kpi_card(label, value, col):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------
df, cust_risk = load_data()

st.sidebar.title("🛡️ SafeBank")
st.sidebar.caption("Transaction Anomaly Detection Platform")
st.sidebar.markdown("---")

min_date, max_date = df["Transaction_Date"].min().date(), df["Transaction_Date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(max_date - pd.Timedelta(days=30), max_date),
                                    min_value=min_date, max_value=max_date)

risk_tiers = st.sidebar.multiselect("Risk tier", options=["Low", "Medium", "High", "Critical"],
                                     default=["Low", "Medium", "High", "Critical"])

locations = st.sidebar.multiselect("Location", options=sorted(df["Location"].unique()), default=[])
merchant_cats = st.sidebar.multiselect("Merchant category", options=sorted(df["Merchant_Category"].unique()), default=[])

page = st.sidebar.radio("Navigate", [
    "📊 Transaction Monitoring", "🔍 Anomaly Visualization", "⚠️ Risk Analysis", "⚡ Real-Time Simulation"
])

# Apply filters
filtered = df.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    filtered = filtered[
        (filtered["Transaction_Date"].dt.date >= date_range[0]) &
        (filtered["Transaction_Date"].dt.date <= date_range[1])
    ]
if risk_tiers:
    filtered = filtered[filtered["Risk_Tier"].isin(risk_tiers)]
if locations:
    filtered = filtered[filtered["Location"].isin(locations)]
if merchant_cats:
    filtered = filtered[filtered["Merchant_Category"].isin(merchant_cats)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered):,}** of {len(df):,} transactions")


# ---------------------------------------------------------------------
# PAGE 1: Transaction Monitoring
# ---------------------------------------------------------------------
if page == "📊 Transaction Monitoring":
    st.title("Transaction Monitoring")
    st.caption("Live overview of banking transaction activity and fraud exposure")

    total_txns = len(filtered)
    suspicious_txns = (filtered["Risk_Tier"].isin(["High", "Critical"])).sum()
    high_risk_customers = cust_risk[cust_risk["Customer_Risk_Tier"].isin(["High", "Critical"])].shape[0]
    fraud_pct = filtered["Fraud_Label"].mean() * 100 if "Fraud_Label" in filtered.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)
    kpi_card("Total Transactions", f"{total_txns:,}", c1)
    kpi_card("Suspicious Transactions", f"{suspicious_txns:,}", c2)
    kpi_card("High-Risk Customers", f"{high_risk_customers:,}", c3)
    kpi_card("Fraud Rate", f"{fraud_pct:.2f}%", c4)

    st.markdown("###")
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Daily Transaction Volume & Fraud Trend")
        daily = filtered.set_index("Transaction_Date").resample("D").agg(
            Total=("Transaction_ID", "count"), Fraud=("Fraud_Label", "sum")
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["Transaction_Date"], y=daily["Total"], name="Total Transactions",
                                  line=dict(color="#2563eb")))
        fig.add_trace(go.Bar(x=daily["Transaction_Date"], y=daily["Fraud"], name="Fraud Transactions",
                              marker_color="#dc2626", yaxis="y2", opacity=0.6))
        fig.update_layout(
            yaxis=dict(title="Total Transactions"),
            yaxis2=dict(title="Fraud Count", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1), height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Risk Tier Breakdown")
        tier_counts = filtered["Risk_Tier"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
        fig2 = px.pie(values=tier_counts.values, names=tier_counts.index, hole=0.5,
                      color=tier_counts.index,
                      color_discrete_map={"Low": "#22c55e", "Medium": "#eab308", "High": "#f97316", "Critical": "#ef4444"})
        fig2.update_layout(height=420, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Transaction Type & Merchant Category Volume")
    c1, c2 = st.columns(2)
    with c1:
        type_counts = filtered["Transaction_Type"].value_counts()
        fig3 = px.bar(x=type_counts.values, y=type_counts.index, orientation="h",
                      labels={"x": "Count", "y": "Transaction Type"}, color=type_counts.values,
                      color_continuous_scale="Blues")
        fig3.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        cat_counts = filtered["Merchant_Category"].value_counts()
        fig4 = px.bar(x=cat_counts.values, y=cat_counts.index, orientation="h",
                      labels={"x": "Count", "y": "Merchant Category"}, color=cat_counts.values,
                      color_continuous_scale="Purples")
        fig4.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 2: Anomaly Visualization
# ---------------------------------------------------------------------
elif page == "🔍 Anomaly Visualization":
    st.title("Anomaly Visualization")
    st.caption("Risk score distributions, anomaly clustering, and geographic risk patterns")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(filtered, x="Risk_Score", nbins=50, color="Risk_Tier",
                           color_discrete_map={"Low": "#22c55e", "Medium": "#eab308", "High": "#f97316", "Critical": "#ef4444"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Fraud Probability vs Anomaly Score")
        sample = filtered.sample(min(5000, len(filtered)), random_state=1)
        fig2 = px.scatter(sample, x="Fraud_Probability_Scaled", y="Anomaly_Score_Scaled",
                          color="Risk_Tier", size="Transaction_Amount", opacity=0.6,
                          color_discrete_map={"Low": "#22c55e", "Medium": "#eab308", "High": "#f97316", "Critical": "#ef4444"},
                          labels={"Fraud_Probability_Scaled": "Fraud Probability (0-100)",
                                  "Anomaly_Score_Scaled": "Anomaly Score (0-100)"})
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Transaction Amount vs Risk Score (Anomaly Scatter)")
    sample2 = filtered.sample(min(6000, len(filtered)), random_state=2)
    fig3 = px.scatter(sample2, x="Transaction_Amount", y="Risk_Score", color="Risk_Tier",
                      hover_data=["Customer_ID", "Merchant_Category", "Location"],
                      color_discrete_map={"Low": "#22c55e", "Medium": "#eab308", "High": "#f97316", "Critical": "#ef4444"})
    fig3.update_layout(height=450)
    st.plotly_chart(fig3, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Risk Heatmap: Hour vs Day of Week")
        heat = filtered.copy()
        heat["Hour"] = heat["Transaction_Date"].dt.hour
        heat["DOW"] = heat["Transaction_Date"].dt.day_name()
        pivot = heat.pivot_table(index="DOW", columns="Hour", values="Risk_Score", aggfunc="mean")
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot.reindex(dow_order)
        fig4 = px.imshow(pivot, color_continuous_scale="Reds", aspect="auto",
                         labels=dict(color="Avg Risk Score"))
        fig4.update_layout(height=380)
        st.plotly_chart(fig4, use_container_width=True)

    with c4:
        st.subheader("Geographic Risk Distribution")
        geo = filtered.groupby("Location").agg(
            Avg_Risk=("Risk_Score", "mean"), Txn_Count=("Transaction_ID", "count")
        ).reset_index().sort_values("Avg_Risk", ascending=False)
        fig5 = px.bar(geo, x="Location", y="Avg_Risk", color="Avg_Risk",
                     color_continuous_scale="Reds", labels={"Avg_Risk": "Average Risk Score"})
        fig5.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 3: Risk Analysis
# ---------------------------------------------------------------------
elif page == "⚠️ Risk Analysis":
    st.title("Risk Analysis")
    st.caption("Fraud probability indicators, customer risk profiles, and live alerts")

    tab1, tab2, tab3 = st.tabs(["High-Risk Alerts", "Customer Risk Profiles", "Fraud Probability Explorer"])

    with tab1:
        st.subheader("🚨 High-Risk Transaction Alerts")
        alerts = filtered[filtered["Risk_Tier"].isin(["High", "Critical"])].sort_values(
            "Risk_Score", ascending=False
        ).head(200)

        def tier_badge(t):
            return {"Critical": "🔴 Critical", "High": "🟠 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}[t]

        display_cols = ["Transaction_ID", "Customer_ID", "Transaction_Amount", "Merchant_Category",
                        "Location", "Device_Type", "Risk_Score", "Risk_Tier", "Transaction_Date"]
        show_df = alerts[display_cols].copy()
        show_df["Risk_Tier"] = show_df["Risk_Tier"].apply(tier_badge)
        st.dataframe(show_df, use_container_width=True, height=460)
        st.download_button("⬇️ Download Alerts (CSV)", show_df.to_csv(index=False), "high_risk_alerts.csv")

    with tab2:
        st.subheader("Customer Risk Profiles")
        c1, c2 = st.columns([1, 2])
        with c1:
            tier_dist = cust_risk["Customer_Risk_Tier"].value_counts()
            fig = px.pie(values=tier_dist.values, names=tier_dist.index, hole=0.5,
                        color=tier_dist.index,
                        color_discrete_map={"Low": "#22c55e", "Medium": "#eab308", "High": "#f97316", "Critical": "#ef4444"})
            fig.update_layout(title="Customer Risk Tier Distribution", height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            top_risk_cust = cust_risk.sort_values("Avg_Risk_Score", ascending=False).head(20)
            fig2 = px.bar(top_risk_cust, x="Avg_Risk_Score", y="Customer_ID", orientation="h",
                         color="Avg_Risk_Score", color_continuous_scale="Reds",
                         labels={"Avg_Risk_Score": "Average Risk Score"})
            fig2.update_layout(title="Top 20 Highest-Risk Customers", height=350, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(cust_risk.sort_values("Avg_Risk_Score", ascending=False), use_container_width=True, height=350)

    with tab3:
        st.subheader("Fraud Probability Explorer")
        selected_txn = st.selectbox("Select a Transaction ID", options=filtered["Transaction_ID"].head(500).tolist())
        row = filtered[filtered["Transaction_ID"] == selected_txn].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud Probability", f"{row['Fraud_Probability']*100:.2f}%")
        c2.metric("Anomaly Score (scaled)", f"{row['Anomaly_Score_Scaled']:.1f}/100")
        c3.metric("Final Risk Score", f"{row['Risk_Score']:.1f}/100", delta=row["Risk_Tier"])

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=row["Risk_Score"],
            title={"text": f"Risk Score - {selected_txn}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1e293b"},
                "steps": [
                    {"range": [0, 35], "color": "#bbf7d0"},
                    {"range": [35, 60], "color": "#fef08a"},
                    {"range": [60, 80], "color": "#fed7aa"},
                    {"range": [80, 100], "color": "#fecaca"},
                ],
            }
        ))
        gauge.update_layout(height=350)
        st.plotly_chart(gauge, use_container_width=True)

        st.json({
            "Customer_ID": row["Customer_ID"],
            "Transaction_Amount": float(row["Transaction_Amount"]),
            "Account_Balance": float(row["Account_Balance"]),
            "Merchant_Category": row["Merchant_Category"],
            "Location": row["Location"],
            "Device_Type": row["Device_Type"],
            "Transaction_Date": str(row["Transaction_Date"]),
        })


# ---------------------------------------------------------------------
# PAGE 4: Real-Time Simulation
# ---------------------------------------------------------------------
elif page == "⚡ Real-Time Simulation":
    st.title("Real-Time Transaction Simulation")
    st.caption("Simulates a live transaction stream being scored by the platform in real time")

    clf, iso_forest, iso_scaler, best_name = load_scoring_models()
    st.info(f"Live scoring uses **{best_name}** (supervised) blended with **Isolation Forest** (anomaly).")

    if "sim_log" not in st.session_state:
        st.session_state.sim_log = []

    c1, c2 = st.columns([1, 3])
    with c1:
        n_stream = st.slider("Transactions per run", 5, 50, 15)
        run_sim = st.button("▶️ Start Simulation", type="primary")
        clear_log = st.button("🗑️ Clear Log")
        if clear_log:
            st.session_state.sim_log = []

    placeholder = st.empty()

    if run_sim:
        from src.train_anomaly_models import ANOMALY_FEATURES
        sample_pool = df.sample(n_stream * 3, random_state=np.random.randint(0, 100000))

        progress = st.progress(0)
        for i, (_, row) in enumerate(sample_pool.head(n_stream).iterrows()):
            X_row = row[FEATURE_COLS].fillna(0).values.reshape(1, -1)
            fraud_proba = clf.predict_proba(X_row)[0, 1]

            X_anom = row[ANOMALY_FEATURES].fillna(0).values.reshape(1, -1)
            X_anom_scaled = iso_scaler.transform(X_anom)
            anomaly_raw = -iso_forest.decision_function(X_anom_scaled)[0]

            risk_score = float(np.clip(fraud_proba * 100 * 0.6 + min(anomaly_raw * 100, 100) * 0.4, 0, 100))
            tier = assign_risk_tier(risk_score)

            entry = {
                "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                "Transaction_ID": row["Transaction_ID"],
                "Customer_ID": row["Customer_ID"],
                "Amount": round(float(row["Transaction_Amount"]), 2),
                "Location": row["Location"],
                "Risk_Score": round(risk_score, 1),
                "Risk_Tier": tier,
            }
            st.session_state.sim_log.insert(0, entry)
            progress.progress((i + 1) / n_stream)
            time.sleep(0.15)

        progress.empty()

    with placeholder.container():
        if st.session_state.sim_log:
            log_df = pd.DataFrame(st.session_state.sim_log)

            latest = log_df.iloc[0]
            alert_col = st.container()
            if latest["Risk_Tier"] in ["High", "Critical"]:
                alert_col.error(
                    f"🚨 ALERT: Transaction {latest['Transaction_ID']} from {latest['Customer_ID']} "
                    f"flagged as **{latest['Risk_Tier']}** risk (score: {latest['Risk_Score']})"
                )
            else:
                alert_col.success(f"✅ Latest transaction {latest['Transaction_ID']} scored normally.")

            def color_tier(val):
                colors = {"Critical": "background-color:#fecaca", "High": "background-color:#fed7aa",
                          "Medium": "background-color:#fef08a", "Low": "background-color:#bbf7d0"}
                return colors.get(val, "")

            st.dataframe(
                log_df.style.applymap(color_tier, subset=["Risk_Tier"]),
                use_container_width=True, height=500
            )
        else:
            st.info("Click **Start Simulation** to stream and score live transactions.")

st.markdown("---")
st.caption("SafeBank Transaction Anomaly Detection Platform · Built with Streamlit, scikit-learn & XGBoost · Demo data is synthetic")
