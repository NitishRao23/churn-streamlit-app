# data_ingestion/ingest.py

import pandas as pd
import os

# Constants
SOURCE_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"  # Place this file manually from Kaggle
RAW_DATA_DIR = "raw_data"
BATCH_SIZE = 100  # Simulate 100 records/day

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def simulate_batch_ingestion():
    # Load the full dataset
    df = pd.read_csv(SOURCE_FILE)

    # Shuffle data to simulate randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into daily batches
    total_rows = len(df)
    num_batches = (total_rows // BATCH_SIZE) + 1

    for i in range(num_batches):
        batch_df = df.iloc[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        batch_file = os.path.join(RAW_DATA_DIR, f"day_{i+1}.csv")
        batch_df.to_csv(batch_file, index=False)
        print(f"[+] Created batch file: {batch_file}")

if __name__ == "__main__":
    create_directory(RAW_DATA_DIR)
    simulate_batch_ingestion()
