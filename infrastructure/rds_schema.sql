-- rds_schema.sql: SCRIPT TO BE RUN IN AWS RDS POSTGRESQL

-- 1. Create the final table for clean data and features
CREATE TABLE clean_customer_features (
    -- IDENTIFIERS & DEMOGRAPHICS
    customerID VARCHAR(10) PRIMARY KEY, -- Unique ID
    gender VARCHAR(10),
    SeniorCitizen INTEGER,
    Partner VARCHAR(5),
    Dependents VARCHAR(5),
    
    -- ACCOUNT & CHARGES
    tenure INTEGER,                      -- Months with the company
    Contract VARCHAR(20),                
    PaymentMethod VARCHAR(30),
    PaperlessBilling VARCHAR(5),
    MonthlyCharges NUMERIC(10, 2),       -- Financial amount
    TotalCharges NUMERIC(10, 2),         -- CLEANED and corrected TotalCharges

    -- SERVICE SUBSCRIPTIONS
    PhoneService VARCHAR(5),
    MultipleLines VARCHAR(20),
    InternetService VARCHAR(20),
    OnlineSecurity VARCHAR(5),
    OnlineBackup VARCHAR(5),
    DeviceProtection VARCHAR(5),
    TechSupport VARCHAR(5),
    StreamingTV VARCHAR(5),
    StreamingMovies VARCHAR(5),
    
    -- ENGINEERED FEATURES & TARGETS
    Recent_Ticket_Count INTEGER,         -- FEATURE 1 (Behavioral Insight)
    Usage_Decline_Pct NUMERIC(4, 2),     -- FEATURE 2 (Behavioral Insight)
    Churn_Numeric INTEGER,               -- Target variable (0/1)
    Churn VARCHAR(5),

    -- METADATA (Optional but good for production proof)
    etl_load_timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- 2. Create an index on the churn column to speed up analysis
CREATE INDEX idx_churn_numeric ON clean_customer_features (Churn_Numeric);