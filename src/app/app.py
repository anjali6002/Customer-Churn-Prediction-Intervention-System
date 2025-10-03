# --- COMPLETE app.py FILE (FINAL VERSION) ---

# --- Part 1: Imports and Setup ---

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import xgboost as xgb 
import sqlite3 # NEW IMPORT

# --- File Paths ---
MODEL_PATH = 'churn_prediction_model.pkl'
FEATURE_LIST_PATH = 'model_feature_list.pkl'
DB_PATH = 'project_db.sqlite'
TABLE_NAME = 'clean_customer_features'

# --- Load the saved artifacts ---
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(FEATURE_LIST_PATH, 'rb') as f:
        feature_list = pickle.load(f)
except FileNotFoundError:
    st.error(f"Error: Required files not found. Check paths: {MODEL_PATH}, {FEATURE_LIST_PATH}")
    st.stop()

# --- Part 2: Database and Data Loading Function (NEW) ---

def load_real_customer_data(customer_id):
    """Loads a single customer record from the local SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Query the clean features table
        query = f"SELECT * FROM {TABLE_NAME} WHERE customerID = '{customer_id}'"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except sqlite3.OperationalError:
        st.error(f"Database error: Could not connect to {DB_PATH}. Run 'python load_to_sqlite.py' first.")
        return pd.DataFrame()


def get_random_customer_id():
    """Gets a random customer ID for the initial view."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Select a random ID from the database
        initial_id = pd.read_sql(f"SELECT customerID FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT 1", conn).iloc[0, 0]
        conn.close()
        return initial_id
    except Exception:
        return '9999-XYZ' # Fallback ID


# --- Part 3: The Core Prediction Function (UNCHANGED LOGIC) ---

def predict_single_customer(customer_data: pd.DataFrame, model, feature_list):
    
    df_single = customer_data.copy()

    # 1. Impute/Clean TotalCharges (Rule from your ETL)
    df_single['TotalCharges'] = df_single['TotalCharges'].replace(' ', 0).astype(float)
    
    # 2. Drop non-feature columns
    df_single = df_single.drop(columns=['customerID', 'Churn'], errors='ignore')

    # 3. Define all object columns for encoding
    categorical_cols = df_single.select_dtypes(include=['object']).columns.tolist()

    # 4. ONE-HOT ENCODING
    df_encoded = pd.get_dummies(df_single, columns=categorical_cols, drop_first=True, dtype=int)
    
    # 5. ALIGN COLUMNS (The FIX for the XGBoost Error)
    df_aligned = pd.DataFrame(0, index=[0], columns=feature_list)
    
    for col in df_encoded.columns:
        if col in df_aligned.columns:
            df_aligned.loc[0, col] = df_encoded.loc[df_encoded.index[0], col]
            
    if 'Churn_Numeric' in df_aligned.columns:
        df_aligned = df_aligned.drop(columns=['Churn_Numeric'])

    # 6. PREDICT
    # Use .values.reshape(1, -1) to ensure the single row is correctly passed
    prediction_proba = model.predict_proba(df_aligned.values.reshape(1, -1))[:, 1][0]
    
    return prediction_proba, df_aligned


# --- Part 4: Streamlit Interface (Modified to use DB) ---

st.set_page_config(layout="wide")
st.title("Customer Churn Intervention System")


# --- Main App Logic ---
st.sidebar.header("Customer Lookup")
# Get a real ID as the default input
customer_id_input = st.sidebar.text_input("Enter Customer ID", value=get_random_customer_id())

if st.sidebar.button("Analyze Risk"):
    
    # LOAD DATA FROM DB
    df_customer = load_real_customer_data(customer_id_input)
    
    if df_customer.empty:
        st.error(f"Customer ID {customer_id_input} not found or database error.")
        st.stop()
        
    # The prediction function needs the data in a DataFrame format
    churn_proba, X_encoded = predict_single_customer(df_customer, model, feature_list)
    risk_score = churn_proba * 100
    
    # --- DISPLAY RISK SCORE ---
    st.header(f"Risk Profile for Customer: {customer_id_input}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Churn Risk Score", value=f"{risk_score:.1f}%")

    # --- ACTION CENTER (Business Logic) ---
    intervention_action = ""
    if risk_score > 85:
        risk_level = "CRITICAL"
        intervention_action = "IMMEDIATE Manager Call & Loyalty Offer (20% Discount)"
        st.error(f"RISK LEVEL: {risk_level}")
    # ... (rest of the action logic) ...
    elif risk_score > 50:
        risk_level = "HIGH"
        intervention_action = "Proactive Support Outreach & Contract Upgrade Offer"
        st.warning(f"RISK LEVEL: {risk_level}")
    else:
        risk_level = "LOW"
        intervention_action = "Standard Check-in Email"
        st.success(f"RISK LEVEL: {risk_level}")
        
    with col2:
        st.subheader("Recommended Intervention")
        st.write(f"**Action:** {intervention_action}")
        
    # --- SHAP EXPLANATION (The WHY) ---
    st.markdown("---")
    st.subheader("Root Cause Analysis (Why the score is high)")
    
    explainer = shap.TreeExplainer(model)
    # Use X_encoded directly for prediction
    shap_values = explainer.shap_values(X_encoded) 

    explanation_df = pd.DataFrame({
        'Feature': X_encoded.columns, 
        'Impact_Score': shap_values[0]
    }).sort_values(by='Impact_Score', ascending=False)
    
    positive_drivers = explanation_df[explanation_df['Impact_Score'] > 0].head(5)
    
    st.table(positive_drivers[['Feature', 'Impact_Score']].rename(columns={'Impact_Score': 'Drives Risk Score Up'}).set_index('Feature'))

    st.markdown("""
    **Intervention Context:** This table shows the features that increased the risk score. 
    The Customer Success Team should address these directly during the call/email.
    """)