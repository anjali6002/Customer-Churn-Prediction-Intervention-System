# --- load_to_sqlite.py: Simulates the "Load" Step to RDS ---
import pandas as pd
import sqlite3

# 1. Load the cleaned data from your export
df_final = pd.read_csv('data/clean/final_customer_data.csv')

# 2. Connect to a local SQLite database (this will create 'project_db.sqlite' if it doesn't exist)
conn = sqlite3.connect('project_db.sqlite')

# 3. Use the schema defined in your rds_schema.sql to structure the data 
# We are creating the clean_customer_features table. if it exists, replace it.
df_final.to_sql('clean_customer_features', conn, if_exists='replace', index=False)

conn.close()

print("Database simulation complete. Data is loaded into 'project_db.sqlite' and ready for the App/Power BI.")