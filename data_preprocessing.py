# data_preprocessing.py
import pandas as pd
import os
import warnings

# === Configuration ===
BASE_DATA_PATH = 'data/'
SALES_FILE = os.path.join(BASE_DATA_PATH, 'sales data-set.csv')
STORES_FILE = os.path.join(BASE_DATA_PATH, 'stores data-set.csv')
FEATURES_FILE = os.path.join(BASE_DATA_PATH, 'Features data set.csv')

OUTPUT_DIR = 'output'
CLEANED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'cleaned_data.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Helper Functions ===


def load_csv_files():
    print("🔄 Loading source CSV files...")
    files_to_load = {
        "sales": SALES_FILE,
        "stores": STORES_FILE,
        "features": FEATURES_FILE
    }
    loaded_dfs = {}
    all_found = True
    for name, path in files_to_load.items():
        if not os.path.exists(path):
            print(f"❌ Error: File missing: {path}")
            all_found = False
        else:
            try:
                loaded_dfs[name + "_df"] = pd.read_csv(path)
                print(f"✅ {name.capitalize()} data loaded from {path}")
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
                all_found = False
    if not all_found:
        exit("One or more source files could not be loaded. Exiting.")
    return loaded_dfs["sales_df"], loaded_dfs["stores_df"], loaded_dfs["features_df"]


def merge_data(sales_df, stores_df, features_df):
    print("🔄 Merging datasets...")
    # Merge sales with features (features_df is the richer source for Date-related features)
    # Suffixes are important if 'IsHoliday' exists in both sales_df and features_df
    df = pd.merge(sales_df, features_df, on=[
                  'Store', 'Date'], how='left', suffixes=('_sales', '_features'))
    # Merge with store information
    df = pd.merge(df, stores_df, on='Store', how='left')
    print(f"✅ Datasets merged. Shape after merge: {df.shape}")
    print(f"Columns after merge: {list(df.columns)}")
    return df


def consolidate_isholiday(df):
    print("🔄 Consolidating 'IsHoliday' columns...")
    sales_col, features_col = 'IsHoliday_sales', 'IsHoliday_features'

    if sales_col in df.columns and features_col in df.columns:
        print(
            f"ℹ️ Found '{sales_col}' and '{features_col}'. Consolidating into 'IsHoliday'.")
        # Combine, treating NaN as False for the OR operation
        df['IsHoliday'] = df[sales_col].fillna(
            False) | df[features_col].fillna(False)
        df.drop(columns=[sales_col, features_col], inplace=True)
    elif sales_col in df.columns:
        print(f"ℹ️ Found only '{sales_col}'. Renaming to 'IsHoliday'.")
        df.rename(columns={sales_col: 'IsHoliday'}, inplace=True)
    elif features_col in df.columns:
        print(f"ℹ️ Found only '{features_col}'. Renaming to 'IsHoliday'.")
        df.rename(columns={features_col: 'IsHoliday'}, inplace=True)
    else:
        print("⚠️ Warning: Neither 'IsHoliday_sales' nor 'IsHoliday_features' found. 'IsHoliday' column might be missing or already consolidated.")

    if 'IsHoliday' in df.columns:
        df['IsHoliday'] = df['IsHoliday'].astype(bool).astype(int)
        print("✅ 'IsHoliday' column processed and converted to int (0 or 1).")
    else:
        print("❌ Error: 'IsHoliday' column could not be created/found after consolidation attempt. This might cause issues downstream.")
        # Optionally, create a default if it's absolutely critical and missing
        # df['IsHoliday'] = 0
        # print("⚠️ Defaulted 'IsHoliday' to 0 as it was missing.")
    return df


def clean_and_convert(df):
    print("🔄 Converting 'Date' column and handling invalid dates...")
    # Assuming date format in 'Features data set.csv' and 'sales data-set.csv' is dayfirst (e.g., DD/MM/YYYY)
    # The file 'Features data set.csv' has dates like '05/02/2010'
    df['Date'] = pd.to_datetime(
        df['Date'], format='%d/%m/%Y', errors='coerce')  # More specific format

    invalid_dates_count = df['Date'].isnull().sum()
    if invalid_dates_count > 0:
        print(
            f"⚠️ Found {invalid_dates_count} rows with invalid/unparseable dates. These rows will be removed.")
        df.dropna(subset=['Date'], inplace=True)
        print(
            f"✅ Removed rows with invalid dates. Shape after removal: {df.shape}")
    else:
        print("✅ 'Date' column successfully converted to datetime. No invalid dates found.")
    return df


def impute_missing(df):
    print("🔄 Imputing missing values...")
    # Size is usually not missing or handled by merge
    numeric_cols_median_impute = ['Temperature',
                                  'Fuel_Price', 'CPI', 'Unemployment']

    if 'Size' in df.columns:  # Size comes from stores_df, should be complete after merge
        if df['Size'].isnull().any():
            print(
                f"⚠️ Missing values found in 'Size'. Imputing with median: {df['Size'].median()}")
            df['Size'] = df['Size'].fillna(df['Size'].median())

    for col in numeric_cols_median_impute:
        if col in df.columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                print(
                    f"ℹ️ Missing values found in '{col}'. Imputing with median: {median_val}")
                df[col] = df[col].fillna(median_val)
            else:
                print(f"✅ No missing values in '{col}'.")
        else:
            print(
                f"⚠️ Warning: Expected numeric column '{col}' not found for imputation.")

    markdown_cols = ['MarkDown1', 'MarkDown2',
                     'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_cols:
        if col in df.columns:
            if df[col].isnull().any():
                print(f"ℹ️ Missing values found in '{col}'. Imputing with 0.")
                df[col] = df[col].fillna(0)
            else:
                print(f"✅ No missing values in '{col}'.")

        else:
            # MarkDowns might not exist in all versions of the dataset or if features_df is minimal
            print(
                f"ℹ️ Note: MarkDown column '{col}' not found. Skipping imputation for it.")

    if 'Weekly_Sales' in df.columns:
        if df['Weekly_Sales'].isnull().any():
            print("ℹ️ Missing values found in 'Weekly_Sales'. Imputing with 0.")
            df['Weekly_Sales'] = df['Weekly_Sales'].fillna(
                0)  # Or median: df['Weekly_Sales'].median()
        else:
            print("✅ No missing values in 'Weekly_Sales'.")
    else:
        print("⚠️ Warning: 'Weekly_Sales' column not found for imputation.")

    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"✅ Dropped {rows_dropped} duplicate rows.")
    else:
        print("✅ No duplicate rows found.")

    return df


def save_cleaned_data(df, file_path):
    print(f"💾 Saving cleaned data to {file_path}...")
    try:
        df.to_csv(file_path, index=False)
        print(
            f"✅ Cleaned data saved successfully. Shape: {df.shape}, Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error saving cleaned data to {file_path}: {e}")

# === Main Flow ===


def main():
    print("🚀 Starting data preprocessing script...")
    sales_df, stores_df, features_df = load_csv_files()

    df_merged = merge_data(sales_df, stores_df, features_df)
    df_holidays_consolidated = consolidate_isholiday(df_merged)
    df_dates_cleaned = clean_and_convert(df_holidays_consolidated)
    df_imputed = impute_missing(df_dates_cleaned)

    print(f"📊 Final cleaned dataset shape before saving: {df_imputed.shape}")
    save_cleaned_data(df_imputed, CLEANED_OUTPUT_FILE)
    print("🎉 Data preprocessing script finished successfully.")


if __name__ == "__main__":
    main()
