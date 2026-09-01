"""
Module 13 — Streamlit Dashboard
SaaS Customer Retention Intelligence System

Run locally with:  streamlit run app.py
(Run from inside the app/ folder, or from the repo root — paths below
resolve relative to this file's location either way.)

Expects the following files to exist in ../data/processed/ (relative to
this script): account_view_engineered.csv, retention_table.csv,
account_risk_scores.csv, account_revenue_risk.csv,
account_recommendations.csv, shap_values_full.csv, model_comparison.csv.
Run Modules 1–12 first (or download the processed CSVs from Drive into
data/processed/) before running this app.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SaaS Retention Intelligence",
    page_icon="📊",
    layout="wide",
)

sns.set_style("whitegrid")

# ── Paths (resolve relative to this script, works regardless of cwd) ─────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(APP_DIR, "..", "data", "processed")


def data_path(filename):
    return os.path.join(PROCESSED_DIR, filename)


# ── Data loading (cached so navigating pages doesn't re-read CSVs) ───────
@st.cache_data
def load_data():
    missing = []
    required = [
        "account_view_engineered.csv", "retention_table.csv",
        "account_revenue_risk.csv", "account_recommendations.csv",
        "shap_values_full.csv", "model_comparison.csv",
    ]
    for f in required:
        if not os.path.exists(data_path(f)):
            missing.append(f)
    if missing:
        return None, missing

    account_view = pd.read_csv(data_path("account_view_engineered.csv"), parse_dates=["signup_date"])
    retention_table = pd.read_csv(data_path("retention_table.csv"), index_col=0)
    retention_table.columns = retention_table.columns.astype(int)
    revenue_risk = pd.read_csv(data_path("account_revenue_risk.csv"))
    recommendations = pd.read_csv(data_path("account_recommendations.csv"))
    shap_values = pd.read_csv(data_path("shap_values_full.csv"))
    model_comparison = pd.read_csv(data_path("model_comparison.csv"), index_col=0)

    return {
        "account_view": account_view,
        "retention_table": retention_table,
        "revenue_risk": revenue_risk,
        "recommendations": recommendations,
        "shap_values": shap_values,
        "model_comparison": model_comparison,
    }, []


data, missing_files = load_data()

if data is None:
    st.error(
        "Missing required data files: " + ", ".join(missing_files) +
        "\n\nRun Modules 1 through 12 first (or copy the processed CSVs "
        "from Google Drive's data/processed/ folder into this repo's "
        "data/processed/ folder) before running this dashboard."
    )
    st.stop()

revenue_risk = data["revenue_risk"]
retention_table = data["retention_table"]
recommendations = data["recommendations"]
shap_values = data["shap_values"]
model_comparison = data["model_comparison"]

# ── Sidebar navigation ────────────────────────────────────────────────────
st.sidebar.title("📊 Retention Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Cohort Analysis", "Customer Risk"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "SaaS Customer Retention Intelligence System — built on the "
    "RavenStack dataset. Predictions are risk-weighted estimates, "
    "not guarantees. See the README for known limitations."
)

# ═══════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("SaaS Retention Intelligence — Overview")

    total_customers = len(revenue_risk)
    churn_rate = revenue_risk["churn_flag"].mean()
    high_risk_count = (revenue_risk["risk_level"] == "High").sum()
    total_mrr_at_risk = revenue_risk["expected_mrr_at_risk"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churn Rate", f"{churn_rate:.1%}")
    col3.metric("High-Risk Customers", f"{high_risk_count:,}")
    col4.metric("Est. Monthly Revenue at Risk", f"${total_mrr_at_risk:,.0f}")

    st.caption(
        "⚠️ Revenue-at-risk is a probability-weighted estimate for "
        "prioritization, not a literal forecast — see Module 10's "
        "calibration note in the README."
    )

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Churn Distribution")
        churn_counts = revenue_risk["churn_flag"].value_counts().sort_index()
        churn_counts.index = ["Retained", "Churned"]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(churn_counts.index, churn_counts.values, color=["#2E7D32", "#C62828"])
        ax.set_ylabel("Accounts")
        st.pyplot(fig)

    with col_right:
        st.subheader("Revenue at Risk by Risk Tier")
        risk_rev = revenue_risk.groupby("risk_level")["expected_mrr_at_risk"].sum().reindex(["Low", "Medium", "High"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(risk_rev.index, risk_rev.values, color=["#2E7D32", "#F9A825", "#C62828"])
        ax.set_ylabel("Expected MRR at Risk ($)")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Top Global Churn Drivers (SHAP)")

    shap_feature_cols = [c for c in shap_values.columns if c not in ["account_id", "top_risk_increasing", "top_risk_decreasing"]]
    mean_abs_shap = shap_values[shap_feature_cols].abs().mean().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(mean_abs_shap.index[::-1], mean_abs_shap.values[::-1], color="#2E5C8A")
    ax.set_xlabel("Mean |SHAP value|")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Model Performance (Module 7)")
    st.dataframe(model_comparison.style.format("{:.3f}"), width="stretch")
    st.caption("Logistic Regression was selected for scoring — see Module 7 for full reasoning.")

# ═══════════════════════════════════════════════════════════════════════
# PAGE 2 — COHORT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
elif page == "Cohort Analysis":
    st.title("Cohort & Retention Analysis")

    st.subheader("Retention Heatmap")
    st.caption("% of each signup cohort still active, by months since signup")

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(
        retention_table, annot=True, fmt=".0%", cmap="YlGnBu",
        cbar_kws={"label": "% retained"}, vmin=0, vmax=1, ax=ax,
    )
    ax.set_xlabel("Months Since Signup")
    ax.set_ylabel("Signup Cohort")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Retention Curves")

    fig, ax = plt.subplots(figsize=(11, 6))
    for cohort in retention_table.index:
        ax.plot(retention_table.columns, retention_table.loc[cohort], marker="o", label=str(cohort))
    ax.set_xlabel("Months Since Signup")
    ax.set_ylabel("% of Cohort Still Active")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Cohort", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    st.pyplot(fig)

    st.markdown("---")
    st.info(
        "⚠️ Newer cohorts show lower retention at matched checkpoints, but "
        "haven't had as much time to be observed as older cohorts — read "
        "this as a signal worth investigating, not a fully settled trend. "
        "See Module 3's Key Findings for the full caveat."
    )

# ═══════════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER RISK
# ═══════════════════════════════════════════════════════════════════════
elif page == "Customer Risk":
    st.title("Customer Risk Lookup")

    account_ids = sorted(revenue_risk["account_id"].unique().tolist())
    selected_id = st.selectbox("Select an Account ID", account_ids)

    account_row = revenue_risk[revenue_risk["account_id"] == selected_id].iloc[0]
    rec_row = recommendations[recommendations["account_id"] == selected_id].iloc[0]

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Churn Probability", f"{account_row['churn_probability']:.1%}")
    col2.metric("Risk Level", account_row["risk_level"])
    col3.metric("Monthly Revenue (MRR)", f"${account_row['mrr_amount']:,.0f}")
    col4.metric("Expected Revenue at Risk", f"${account_row['expected_mrr_at_risk']:,.0f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Account Details")
        st.write(f"**Plan Tier:** {account_row['plan_tier']}")
        st.write(f"**Industry:** {account_row['industry']}")
        st.write(f"**Billing Frequency:** {account_row['billing_frequency']}")
        st.write(f"**Current Status:** {'Churned' if account_row['churn_flag'] else 'Active'}")

    with col_right:
        st.subheader("Suggested Action")
        st.write(f"**Category:** {rec_row['action_category']}")
        st.write(f"**Urgency:** {rec_row['urgency']}")
        st.write(rec_row["suggested_action"])
        st.caption(
            "⚠️ This is a suggested next step based on correlated risk "
            "factors, not a proven causal fix — see Module 12."
        )

    st.markdown("---")
    st.subheader("Why This Score? (SHAP Explanation)")

    shap_row = shap_values[shap_values["account_id"] == selected_id]
    if not shap_row.empty:
        shap_feature_cols = [c for c in shap_values.columns if c not in ["account_id", "top_risk_increasing", "top_risk_decreasing"]]
        row_values = shap_row[shap_feature_cols].iloc[0].astype(float).sort_values()

        top_n = 10
        display_values = pd.concat([row_values.head(top_n // 2), row_values.tail(top_n // 2)])

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#C62828" if v > 0 else "#2E7D32" for v in display_values.values]
        ax.barh(display_values.index, display_values.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (red = increases risk, green = decreases risk)")
        st.pyplot(fig)
    else:
        st.warning("No SHAP explanation available for this account.")

st.sidebar.markdown("---")
st.sidebar.caption("Module 13 — SaaS Customer Retention Intelligence System")
