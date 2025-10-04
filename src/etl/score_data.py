# --- score_data.py (Run this once to generate the score column) ---

import pandas as pd
import pickle
import os

# --- Path Resolution (Assumes running from project root) ---
ROOT_DIR = os.getcwd()
MODEL_PATH = os.path.join(ROOT_DIR, 'src', 'app', 'churn_prediction_model.pkl')
FEATURE_LIST_PATH = os.path.join(ROOT_DIR, 'src', 'app', 'model_feature_list.pkl')
INPUT_CSV_PATH = os.path.join(ROOT_DIR, 'data', 'clean', 'final_customer_data.csv')
OUTPUT_CSV_PATH = os.path.join(ROOT_DIR, 'data', 'clean', 'scored_customer_data.csv')

print(f"Loading data from: {INPUT_CSV_PATH}")

# --- 1. Load Artifacts ---
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURE_LIST_PATH, 'rb') as f:
        feature_list = pickle.load(f)
except FileNotFoundError as e:
    print(f"ERROR: Model or Feature list not found. Detail: {e}")
    exit()

# Define the final feature list used by the model
FINAL_FEATURE_COLUMNS = [f for f in feature_list if f != 'Churn_Numeric']

# --- 2. Load and Prepare Data (Adaptation of Streamlit logic) ---
df = pd.read_csv(INPUT_CSV_PATH)
df['TotalCharges'] = df['TotalCharges'].replace(' ', 0).astype(float)

# Save customer ID for joining later
customer_ids = df['customerID']
target_col = df['Churn']
target_num_col = df['Churn_Numeric']

# Drop non-feature columns
df_features = df.drop(columns=['customerID', 'Churn', 'Churn_Numeric'], errors='ignore')

# ONE-HOT ENCODING (Similar to Streamlit app)
categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True, dtype=int)

# ALIGN COLUMNS to ensure model input matches training data
df_aligned = pd.DataFrame(0, index=df_encoded.index, columns=feature_list)
for col in df_encoded.columns:
    if col in df_aligned.columns:
        df_aligned[col] = df_encoded[col]

# Remove the target column from the feature set if it exists
df_aligned = df_aligned.drop(columns=['Churn_Numeric'], errors='ignore')

# --- 3. Predict Probabilities ---
X_final = df_aligned[FINAL_FEATURE_COLUMNS]

# Predict probability of churn (index 1)
predicted_probs = model.predict_proba(X_final.values)[:, 1]

# --- 4. Create Final Scored DataFrame ---
df_scored = pd.DataFrame({
    'customerID': customer_ids,
    'Predicted_Prob': predicted_probs
})

# Merge scores back to original data and calculate percentage
df_final_output = df.merge(df_scored, on='customerID')
df_final_output['C_Predicted_Risk_Pct'] = df_final_output['Predicted_Prob'] * 100

# --- 5. Save the New Scored Data ---
df_final_output.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"SUCCESS: Scored data saved to: {OUTPUT_CSV_PATH}")

# --- END score_data.py ---