# --- src/etl/load_to_sqlite.py (FINAL CORRECTED PATHS) ---
import pandas as pd
import sqlite3
import os

# --- ABSOLUTE PATH CALCULATION ---
# 1. Get the directory of the currently executing script (src/etl)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Get the directory of the 'src' folder
SRC_DIR = os.path.dirname(SCRIPT_DIR)

# 3. Get the Project Root Directory (C:\Users\Home\Churn_System_Project)
ROOT_DIR = os.path.dirname(SRC_DIR) 

# Define absolute paths for input and output files
DB_PATH_ABS = os.path.join(ROOT_DIR, 'project_db.sqlite')

# *** PATH CORRECTION BASED ON YOUR STRUCTURE ***
# 'data' is a direct child of the project root, NOT a child of 'src'
CSV_PATH = os.path.join(ROOT_DIR, 'data', 'clean', 'final_customer_data.csv')
# *** END PATH CORRECTION ***
# --- END PATH CALCULATION ---

try:
    # --- DEBUGGING INFORMATION ---
    print(f"DEBUG: Project Root is: {ROOT_DIR}")
    print(f"Attempting to load CSV from: {CSV_PATH}")
    print(f"Connecting to DB at: {DB_PATH_ABS}")
    # --- END DEBUGGING ---
    
    # 1. Load the cleaned data 
    df_final = pd.read_csv(CSV_PATH)
    
    # 2. Connect to the local SQLite database using the ABSOLUTE path
    conn = sqlite3.connect(DB_PATH_ABS)

    # 3. Create the table and load data 
    df_final.to_sql('clean_customer_features', conn, if_exists='replace', index=False)

    conn.close()
    
    print("Database table 'clean_customer_features' successfully created and loaded.")

except FileNotFoundError as e:
    print(f"\nFATAL ERROR: CSV not found.")
    print(f"Detail: {e}")
    print(f"File was expected at: {CSV_PATH}")
except Exception as e:
    print(f"AN UNEXPECTED DATABASE ERROR OCCURRED: {e}")