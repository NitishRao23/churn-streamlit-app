# data_cleaning/clean_transform.py

import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = "raw_data/"
OUTPUT_PATH = "transformed_data/cleaned_telco_data.csv"

def load_all_csvs(path):
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = [pd.read_csv(os.path.join(path, f)) for f in csv_files]
    return pd.concat(df_list, ignore_index=True)

def clean_data(df):
    # Replace spaces with NaN
    df.replace(" ", np.nan, inplace=True)

    # Drop rows with missing TotalCharges
    df.dropna(subset=['TotalCharges'], inplace=True)

    # Convert TotalCharges to float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    # Encode binary columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                   'Churn', 'SeniorCitizen']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    # Fill missing values in other columns with mode
    df.fillna(df.mode().iloc[0], inplace=True)

    # Feature Engineering
    df['TenureGroup'] = pd.cut(df['tenure'],
                               bins=[0, 12, 24, 48, 60, np.inf],
                               labels=['0–12', '13–24', '25–48', '49–60', '60+'])

    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df['TotalServices'] = df[service_cols].apply(lambda row: sum(val == 'Yes' or val == 1 for val in row), axis=1)

    df['IsLoyal'] = df['tenure'].apply(lambda x: 1 if x > 60 else 0)

    return df

def save_transformed_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    df_raw = load_all_csvs(RAW_DATA_PATH)
    df_cleaned = clean_data(df_raw)
    save_transformed_data(df_cleaned, OUTPUT_PATH)
