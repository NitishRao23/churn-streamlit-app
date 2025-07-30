import pandas as pd
from sqlalchemy import create_engine

# Load cleaned data
df = pd.read_csv('transformed_data/cleaned_telco_data.csv')

# PostgreSQL connection details
user = 'postgres'
password = 'Nitish@23'
host = '127.0.0.1'
port = '5432'
database = 'churn_project'

# Create engine
engine = create_engine('postgresql+psycopg2://postgres:Nitish%4023@127.0.0.1:5432/churn_project')


# Load data into PostgreSQL
df.to_sql('telco_churn_data', engine, if_exists='replace', index=False)

print("✅ Data loaded into PostgreSQL successfully!")
