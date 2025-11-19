!pip install https://github.com/mrjbq7/ta-lib/archive/refs/tags/TA_Lib-0.4.24.tar.gz

# Verify installation
try:
    import talib
    print("TA-Lib imported successfully!")
except ImportError:
    print("Import failed, trying alternative solution...")
    !pip install talib-binary

import numpy as np
import pandas as pd
import talib
from google.colab import files

# Read data (adjust based on actual format)
df = pd.read_csv("Brentbase.csv")

# Ensure date column exists
date_col = [col for col in df.columns if 'date' in col.lower()][0]
df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

# Standardize column names
column_mapping = {
    'open': 'Open',
    'high': 'High', 
    'low': 'Low',
    'close': 'Close',
}

for orig, standard in column_mapping.items():
    if orig in df.columns.str.lower():
        df.rename(columns={orig: standard}, inplace=True)

print("Standardized column names:", df.columns)

# Ensure required price columns exist
ohlc_columns = ['Open', 'High', 'Low', 'Close']

if all(col in df.columns for col in ohlc_columns):
    # Original indicators
    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)

    macd, macd_signal, macd_hist = talib.MACD(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist

    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(
        df['Close'], timeperiod=20)

    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)

    df['SlowK'], df['SlowD'] = talib.STOCH(
        df['High'], df['Low'], df['Close'],
        fastk_period=14, slowk_period=3, slowd_period=3
    )

    # Additional technical indicators
    # Average True Range (ATR) - Volatility measure
    df['ATR_14'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Commodity Channel Index (CCI) - Trend strength
    df['CCI_20'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=20)

    # Williams %R - Overbought/Oversold
    df['Williams_%R'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Momentum - Price change speed
    df['Momentum_10'] = talib.MOM(df['Close'], timeperiod=10)

    # Price volatility (standard deviation based)
    df['Volatility_30'] = df['Close'].pct_change().rolling(window=30).std()
    
    # Additional indicators
    df['ADX_14'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ROC_10'] = talib.ROC(df['Close'], timeperiod=10)
    
    print("Technical indicators calculated successfully!")
else:
    missing = [col for col in ohlc_columns if col not in df.columns]
    print(f"Missing required columns: {missing}, cannot calculate indicators")

# Save results
output_file = 'Brent_Technical_Analysis3.csv'
df.to_csv(output_file, encoding='utf-8-sig')
print("Analysis completed and saved to", output_file)
