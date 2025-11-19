import pandas as pd
import numpy as np
import talib

# ===================== Configuration Parameters =====================
DATA_SOURCE = "https://cn.investing.com/commodities/real-time-futures"
INPUT_FILE = "WTIalldata.csv"
OUTPUT_FILE = "processed_WTI_data.csv"
WINDOW_SIZE = 5
DATE_COLUMN = "date"
TARGET_COLUMN = "Close"

# ===================== Data Preprocessing =====================
def load_and_preprocess_data(file_path):
    """Load and preprocess the raw commodity price data"""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Original shape: {df.shape}")
        
        # Validate and convert date column
        if DATE_COLUMN not in df.columns:
            raise ValueError(f"Date column '{DATE_COLUMN}' not found in dataset")
        
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        df = df.sort_values(by=DATE_COLUMN).reset_index(drop=True)
        print("Data sorted chronologically")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error during data loading: {str(e)}")
        return None

def handle_missing_values(df, target_column, window_size):
    """Handle missing values using trailing window mean and forward fill"""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    original_missing = df[target_column].isnull().sum()
    print(f"Original missing values in {target_column}: {original_missing}")
    
    if original_missing > 0:
        # Calculate trailing moving average (using only historical data)
        trailing_ma = (df[target_column]
                      .rolling(window=window_size, min_periods=1, closed="right")
                      .mean()
                      .shift(1))
        
        # Fill missing values with trailing MA, then forward fill as backup
        df[target_column] = (df[target_column]
                            .fillna(trailing_ma)
                            .fillna(method="ffill"))
        
        final_missing = df[target_column].isnull().sum()
        print(f"Missing values after imputation: {final_missing}")
    
    return df
