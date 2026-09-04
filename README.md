# SaaS Customer Retention Intelligence System

An end-to-end machine learning system that predicts **SaaS customer churn**, explains *why* customers are at risk, analyzes retention by signup cohort, and estimates potential recurring revenue at risk.

🔗 **Live Dashboard:** https://saas-retention-intelligence-ezgne9epgwbntfhgzgd5bk.streamlit.app/

Built as a capstone project after completing Andrew Ng's **Machine Learning Specialization**.

---

## Overview

The goal was to go beyond simply predicting churn and build a system that answers:

* Who is likely to churn?
* Why are they at risk?
* How does retention change across signup cohorts?
* How much revenue is potentially exposed?
* What action could a retention team consider?

### Pipeline

```text
Data Understanding → EDA → Cohort Analysis → Feature Engineering
→ ML Models → Evaluation → SHAP → Revenue at Risk → Recommendations → Dashboard
```

## Dataset

**RavenStack** — a synthetic multi-table SaaS dataset containing 500 accounts, including:

* Account & subscription information
* Product feature usage
* Support tickets
* Churn events

The dataset was chosen instead of commonly used Telco/Bank churn datasets to build a less generic portfolio project.

---

## Key Learning: Understanding the Data Matters

The most valuable part of the project wasn't the final model. It was discovering problems in the data that initially made the model look better than it really was.

### Data leakage

A `days_since_last_activity` feature was calculated using each customer's cancellation date for churned customers but a different reference date for active customers.

This created a strong suspicious correlation (**r = −0.55**) with churn.

After correcting the feature to use one consistent reference date, test accuracy dropped from **~92% to ~60–75%**.

The lower number was the trustworthy one.

I also found that some currently active customers had historical churn records because they had previously cancelled and later reactivated.

These issues reinforced one of the biggest lessons from the project:

> **A good ML model starts with understanding the data, not just choosing an algorithm.**

---

## Model Results

Three models were compared using precision, recall, F1, ROC-AUC, and PR-AUC because churn is an imbalanced classification problem.

| Model                   | Precision |    Recall |        F1 |   ROC-AUC |    PR-AUC |
| ----------------------- | --------: | --------: | --------: | --------: | --------: |
| **Logistic Regression** |     0.348 | **0.727** | **0.471** | **0.684** | **0.420** |
| Decision Tree           |     0.231 |     0.409 |     0.295 |     0.474 |     0.241 |
| Random Forest           |     0.200 |     0.045 |     0.074 |     0.599 |     0.276 |

**Logistic Regression** was selected as the final model based on its stronger recall and F1 performance.

---

## Explainability & Revenue at Risk

**SHAP** was used to explain individual predictions and identify the factors increasing or decreasing each customer's churn risk.

A custom **engagement trend ratio** became the strongest global driver, demonstrating the value of feature engineering.

Revenue exposure was estimated using:

```text
Revenue at Risk = MRR × Churn Probability
```

The model estimated **$522,747/month** in probability-weighted revenue at risk. Because predicted probabilities were not perfectly calibrated, this is treated as a **prioritization metric rather than a literal revenue forecast**.

Interestingly, only **1 of the top 10** accounts by revenue exposure also appeared in the top 10 by churn probability — showing why churn probability alone isn't enough for retention prioritization.

---

## Dashboard

The Streamlit dashboard includes:

**Overview**

* Churn KPIs
* Model comparison
* Revenue at risk
* Global SHAP drivers

**Cohort Analysis**

* Retention heatmap
* Retention curves

**Customer Risk**

* Churn probability
* Risk level
* Revenue exposure
* SHAP explanation
* Suggested action

---

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · SHAP · Matplotlib · Seaborn · Streamlit**

---

## Limitations

* Dataset is synthetic
* No individual feature strongly predicts churn
* Some tenure-based features have a documented mild leakage limitation
* Revenue-at-risk is a probability-weighted estimate, not a forecast
* Recommendations are not causally validated


