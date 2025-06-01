import pandas as pd
import numpy as np
import os

file_path = "data/Features data set.csv"  # change to uploaded file path if needed
df = pd.read_csv(file_path)

# 🔽 Force convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 🛑 Remove invalid dates
if df['Date'].isnull().any():
    print("⚠️ Some dates could not be parsed. Dropping them.")
    df = df.dropna(subset=['Date'])

# 🔽 Extract Month, Year, Week
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Week'] = df['Date'].dt.isocalendar().week

# 🔽 Encode Store Type if present
if 'Type' in df.columns:
    df['Store_Type_Code'] = df['Type'].astype('category').cat.codes
else:
    df['Store_Type_Code'] = 0  # default if 'Type' not available

# 🔽 Create dummy columns if missing (to match model)
required_cols = ['Lag_1_Week_Sales', 'Lag_2_Week_Sales', 'Rolling_3_Week_Avg', 'Rolling_5_Week_Avg']
for col in required_cols:
    if col not in df.columns:
        df[col] = 0  # or np.nan if you want to later dropna()

# 🔽 Drop rows with NaNs (if any)
df.dropna(inplace=True)

# 🔽 Save
output_path = "output/preprocessed_uploaded.csv"
os.makedirs("output", exist_ok=True)
df.to_csv(output_path, index=False)

print("✅ Preprocessing for uploaded file complete. File saved to:", output_path)
