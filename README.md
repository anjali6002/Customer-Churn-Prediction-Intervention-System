# Customer-Churn-Prediction-Intervention-System

This repository contains the **Customer Churn Prediction & Intervention System** — an end-to-end advanced analytics and BI solution designed to proactively reduce churn and protect telecom revenue using predictive modeling, explainable AI, and interactive dashboards.


---

## Executive Summary

This project delivers a production-ready system for proactive customer retention, designed to translate complex Machine Learning predictions into measurable business actions. The solution focuses on maximizing revenue protection by identifying and diagnosing high-risk customer behavior within a simulated telecommunications environment.

---

## Core Project Achievements

| Achievement Area       | Key Metric / Insight                       | Portfolio Value |
|------------------------|---------------------------------------------|-----------------|
| **Predictive Accuracy** | **99.92% AUC-ROC**                         | Confirms model reliability, validating feature engineering efforts and exceeding standard benchmarks. |
| **Actionable Insight** | **Usage Decline = 7.3x Impact**             | SHAP analysis confirmed `Usage Decline Percentage` as the primary churn driver, shifting intervention strategy from static demographics to dynamic customer behavior. |
| **Operational Deployment** | **Streamlit Action Center**              | Deployed an application providing real-time risk scores and SHAP explanations for personalized customer outreach. |
| **Business Reporting** | **3-Page Power BI Dashboard**               | Fully executed visualization layer detailing Executive KPIs, Feature Importance, and ROI Simulation. |

---

## Technical Architecture & Data Pipeline

The system is designed around a scalable, cloud-native framework, proving proficiency across the data lifecycle.

**Pipeline Flow:**  
`Data Sources → AWS S3 → (Automated ETL) → RDS PostgreSQL → ML / BI Layer`

### Key Technical Files

| Component            | File Path                           | Function in the Project |
|----------------------|--------------------------------------|--------------------------|
| **Cloud ETL Logic**  | `src/etl/lambda_etl_script.py`       | Python script designed for AWS Lambda to automate data cleansing and feature generation. |
| **Database Schema**  | `infrastructure/rds_schema.sql`      | Defines the final relational table structure (PostgreSQL) for high-performance BI access. |
| **ML Artifact**      | `src/models/churn_prediction_model.pkl` | XGBoost classifier trained with SHAP values, ready for deployment. |
| **Operational App**  | `src/app/app.py`                     | Streamlit application that connects to the database and serves live predictions. |

---

## Visualization & Analysis

### Feature Engineering
High performance was achieved by integrating simulated behavioral data with the core IBM Telco dataset to create novel features, including:
- `Usage_Decline_Pct`
- `Recent_Ticket_Count`

### Power BI Dashboard Structure
The final dashboard is built to serve different organizational needs:

- **Page 1 (Executive Summary):** High-level KPIs (AUC, Revenue at Risk, Churn Forecast).
- **Page 2 (Analytical Deep Dive):** Feature importance plot, confusion matrix, and the strategic CLV vs. Risk Segmentation matrix.
- **Page 3 (Action Center):** Filterable priority call list and a dynamic ROI simulation gauge.

---

## Quick Start Guide (Local Execution)

1. **Clone the repository:**
   ```bash
   git clone [repository-url]
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt


3. **Setup Database:**
   Load the cleaned data into the local SQLite database:
   ```bash
   python src/etl/load_to_sqlite.py


4. **Run Streamlit App:**
   ```bash
   streamlit run src/app/app.py


5. **View Dashboard:**
   Open the .pbix file in Power BI Desktop and connect to the source CSV file.
