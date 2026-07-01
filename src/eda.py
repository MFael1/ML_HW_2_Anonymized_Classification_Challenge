"""
eda.py — Exploratory Data Analysis for Classification Competition.
"""

import pandas as pd
import numpy as np
import os
from config import CONFIG

def run_eda():
    train_path = CONFIG["paths"]["train_csv"]
    
    if not os.path.exists(train_path):
        print(f"Data not found at {train_path}. Make sure it is downloaded.")
        return
        
    df = pd.read_csv(train_path)
    print("=== Shape of Training Data ===")
    print(df.shape)
    
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print("\n=== Target Class Balance ===")
    if "target" in df.columns:
        print(df["target"].value_counts(normalize=True))
    
    print("\n=== Unique Values per Column (Categorical Candidates) ===")
    for col in df.columns:
        unique_cnt = df[col].nunique()
        if unique_cnt < 50:
            print(f"{col}: {unique_cnt} unique values")
            
    print("\n=== Describe (Numerical) ===")
    print(df.describe())

if __name__ == "__main__":
    run_eda()
