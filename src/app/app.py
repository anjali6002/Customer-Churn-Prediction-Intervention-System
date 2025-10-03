# --- COMPLETE app.py FILE (FINAL VERSION - PATHS RESOLVED AND SHAP FIXED) ---

# --- Part 1: Imports and Setup ---

import streamlit as st
import streamlit.components.v1 as components 
import pandas as pd
import numpy as np
import pickle
import shap
import xgboost as xgb
import sqlite3
import os

# --- Helper function for SHAP plots in Streamlit ---
def st_shap(plot, height=None):
    """Utility function to display SHAP plots in Streamlit."""
    if hasattr(plot, 'html'):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
    else:
        st.pyplot(plot) 

# --- Path Resolution for Model Artifacts and DB ---
# This assumes the script is run from the project root: 'streamlit run src/app/app.py'
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up twice to get to project root
ROOT_DIR = os.path.dirname(os.path.dirname(APP_DIR)) 

# Model and Feature paths (relative to app.py)
MODEL_PATH = os.path.join(APP_DIR, 'churn_prediction_model.pkl')
FEATURE_LIST_PATH = os.path.join(APP_DIR, 'model_feature_list.pkl')

# Database path (relative to project root)
DB_PATH_ABS = os.path.join(ROOT_DIR, 'project_db.sqlite')
TABLE_NAME = 'clean_customer_features'

# --- Load the saved artifacts ---
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(FEATURE_LIST_PATH, 'rb') as f:
        feature_list = pickle.load(f)
except FileNotFoundError:
    st.error(f"Error: Required model files not found. Check these paths: {MODEL_PATH}, {FEATURE_LIST_PATH}")
    st.stop()

# Define the final, pure list of features (excluding the target variable)
FINAL_FEATURE_COLUMNS = [f for f in feature_list if f != 'Churn_Numeric']


# --- Part 2: Database and Data Loading Function ---

def load_real_customer_data(customer_id):
    """Loads a single customer record from the local SQLite database."""
    try:
        # Connect to the database using the ABSOLUTE path
        conn = sqlite3.connect(DB_PATH_ABS)
        query = f"SELECT * FROM {TABLE_NAME} WHERE customerID = '{customer_id}'"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except sqlite3.OperationalError as e:
        st.error(f"Database error: Table not found. Please run the load script first. Detail: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to connect or query database. Ensure DB is created. Detail: {e}")
        return pd.DataFrame()


def get_random_customer_id():
    """Gets a random customer ID for the initial view."""
    try:
        conn = sqlite3.connect(DB_PATH_ABS)
        initial_id = pd.read_sql(f"SELECT customerID FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT 1", conn).iloc[0, 0]
        conn.close()
        return initial_id
    except Exception:
        # Fallback if DB isn't ready
        return '8892-FZLPG'


# --- Part 3: The Core Prediction Function ---

def predict_single_customer(customer_data: pd.DataFrame, model, feature_list):

    df_single = customer_data.copy()

    # 1. Impute/Clean TotalCharges
    df_single['TotalCharges'] = df_single['TotalCharges'].replace(' ', 0).astype(float)

    # 2. Drop non-feature columns
    df_single = df_single.drop(columns=['customerID', 'Churn'], errors='ignore')

    # 3. Define all object columns for encoding
    categorical_cols = df_single.select_dtypes(include=['object']).columns.tolist()

    # 4. ONE-HOT ENCODING
    df_encoded = pd.get_dummies(df_single, columns=categorical_cols, drop_first=True, dtype=int)

    # 5. ALIGN COLUMNS
    df_aligned = pd.DataFrame(0, index=[0], columns=feature_list)

    for col in df_encoded.columns:
        if col in df_aligned.columns:
            df_aligned.loc[0, col] = df_encoded.loc[df_encoded.index[0], col]

    # CRITICAL: Filter out the target variable if it exists in the feature list
    if 'Churn_Numeric' in df_aligned.columns:
        df_aligned = df_aligned.drop(columns=['Churn_Numeric'])

    # 6. PREDICT
    X_final = df_aligned[FINAL_FEATURE_COLUMNS]

    # Ensure X_final is a DMatrix if using raw XGBoost model
    if isinstance(model, xgb.Booster):
        d_matrix = xgb.DMatrix(X_final.values.reshape(1, -1))
        prediction_proba = model.predict(d_matrix)[0]
    else:
        # For Scikit-learn or wrapped models
        try:
            prediction_proba = model.predict_proba(X_final.values.reshape(1, -1))[:, 1][0]
        except AttributeError:
             # Fallback for models without predict_proba (shouldn't happen for churn)
             prediction_proba = model.predict(X_final.values.reshape(1, -1))[0]


    return prediction_proba, X_final


# --- Part 4: Streamlit Interface ---

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
        st.stop()

    # The prediction function returns the churn probability and the aligned feature vector
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
    elif risk_score > 50:
        risk_level = "HIGH"
        intervention_action = "Email retention campaign & Special feature trial."
        st.warning(f"RISK LEVEL: {risk_level}")
    elif risk_score > 20:
        risk_level = "MEDIUM"
        intervention_action = "Proactive survey call to understand needs."
        st.info(f"RISK LEVEL: {risk_level}")
    else:
        risk_level = "LOW"
        intervention_action = "Monitor only. No immediate action required."
        st.success(f"RISK LEVEL: {risk_level}")

    with col2:
        st.metric(label="Recommended Intervention", value=intervention_action)

    # --- FEATURE IMPORTANCE/EXPLANATION (SHAP) ---
    st.header("Churn Drivers (SHAP Explanation)")

    # Ensure the feature names are correct for display
    X_display = X_encoded.copy()
    X_display.columns = FINAL_FEATURE_COLUMNS
    
    try:
        explainer = shap.TreeExplainer(model)
        
        # Determine which index to use for the plot (0 or 1)
        all_shap_values = explainer.shap_values(X_encoded)
        shap_values = None
        expected_value = None
        
        # CHECK: If explainer returns a list of two arrays (standard for binary classification)
        if isinstance(all_shap_values, list) and len(all_shap_values) == 2:
            # Use the second array (index 1) for the positive class (Churn)
            shap_values = all_shap_values[1]
            expected_value = explainer.expected_value[1]
        
        # CHECK: If explainer returns a single array (when only one output is calculated)
        elif not isinstance(all_shap_values, list):
            # Use the single array 
            shap_values = all_shap_values
            expected_value = explainer.expected_value 
        
        # CHECK: If explainer returns a list of one array (edge case)
        elif isinstance(all_shap_values, list) and len(all_shap_values) == 1:
             shap_values = all_shap_values[0]
             expected_value = explainer.expected_value[0]
        
        if shap_values is not None:
            # Display the SHAP force plot
            st_shap(shap.force_plot(expected_value, shap_values, X_display))
        else:
             st.warning("SHAP Explainer returned an unexpected format. Cannot plot.")


    except Exception as e:
        st.error(f"Error generating SHAP plot. Ensure 'shap' and 'xgboost' are installed correctly.")
        # Print the detailed error to the terminal for debugging
        print(f"SHAP PLOT CRITICAL ERROR: {e}")

    # --- Raw Data Display ---
    st.markdown("---")
    st.subheader("Customer's Raw Feature Values")
    st.dataframe(df_customer, hide_index=True)