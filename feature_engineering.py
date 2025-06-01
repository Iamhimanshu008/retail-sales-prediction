# feature_engineering.py (VERSION 3 - Improved robustness and clarity)
import pandas as pd
import os
import warnings

# --- Configuration ---
CLEANED_DATA_FILE = 'output/cleaned_data.csv'
FEATURE_ENGINEERED_OUTPUT_FILE = 'output/feature_engineered_data.csv'

# Columns expected by model_training.py (ensure these are created or defaulted)
# This helps ensure schema consistency
EXPECTED_LAG_ROLL_FEATURES = [
    'Lag_1_Week_Sales', 'Lag_2_Week_Sales',
    'Rolling_3_Week_Avg_Sales', 'Rolling_5_Week_Avg_Sales'
]
EXPECTED_TYPE_FEATURE = 'Store_Type_Code'


# --- 1. Load Cleaned Data ---
print("⚙️ Starting feature engineering script...")
print(f"⏳ Loading cleaned data from '{CLEANED_DATA_FILE}'...")
if not os.path.exists(CLEANED_DATA_FILE):
    print(f"❌ Error: Cleaned data file '{CLEANED_DATA_FILE}' not found.")
    print("Please run the data_preprocessing.py script first to generate it.")
    exit()

try:
    df = pd.read_csv(CLEANED_DATA_FILE, parse_dates=['Date'])
    print(
        f"✅ Cleaned data loaded successfully. Shape: {df.shape}, Columns: {list(df.columns)}")
except Exception as e:
    print(f"❌ Error loading cleaned data from '{CLEANED_DATA_FILE}': {e}")
    exit()

# --- Sanity Check & Defaulting for Critical Input Columns ---

# Ensure 'IsHoliday' column exists (should be handled by data_preprocessing.py)
if 'IsHoliday' not in df.columns:
    warnings.warn(
        "Critical Warning: 'IsHoliday' column not found in cleaned_data.csv. "
        "This should have been created by data_preprocessing.py. "
        "Defaulting 'IsHoliday' to 0 (False) for all rows. This may impact model accuracy.",
        UserWarning
    )
    df['IsHoliday'] = 0
else:
    # Ensure it's integer type if it exists
    df['IsHoliday'] = df['IsHoliday'].astype(int)
    print("✅ 'IsHoliday' column found and confirmed as integer.")


# --- 2. Date-based Features ---
print("⏳ Creating date-based features...")
if 'Date' not in df.columns:
    print("❌ Error: 'Date' column not found in loaded data. Cannot create date-based features.")
    print("Ensure 'Date' column is present and correctly parsed in 'cleaned_data.csv'.")
    exit()

try:
    df['DayOfWeek'] = df['Date'].dt.dayofweek    # Monday=0, Sunday=6
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    print(
        f"✅ Date-based features (DayOfWeek, Month, Year, WeekOfYear) created. Current columns: {list(df.columns)}")
except AttributeError as e:
    print(
        f"❌ Error creating date features. Is 'Date' column properly parsed as datetime? Error: {e}")
    exit()


# --- 3. Lag Features for 'Weekly_Sales' ---
print("⏳ Creating lag features for 'Weekly_Sales'...")
if 'Weekly_Sales' not in df.columns:
    warnings.warn(
        "⚠️ 'Weekly_Sales' column not found in input data. "
        "Lag features (Lag_1_Week_Sales, Lag_2_Week_Sales) cannot be calculated from source. "
        "These columns will be created and filled with 0.",
        UserWarning
    )
    df['Lag_1_Week_Sales'] = 0
    df['Lag_2_Week_Sales'] = 0
else:
    if df['Weekly_Sales'].isnull().any():
        warnings.warn(
            "⚠️ 'Weekly_Sales' column contains NaNs. Filling with 0 before creating lag features. Consider addressing NaNs in preprocessing.", UserWarning)
        df['Weekly_Sales'].fillna(0, inplace=True)

    # Crucial for correct lag/rolling calculation
    df = df.sort_values(by=['Store', 'Dept', 'Date'])
    df['Lag_1_Week_Sales'] = df.groupby(['Store', 'Dept'])[
        'Weekly_Sales'].shift(1)
    df['Lag_2_Week_Sales'] = df.groupby(['Store', 'Dept'])[
        'Weekly_Sales'].shift(2)
    print("✅ Lag features for 'Weekly_Sales' created.")

# --- 4. Rolling Average Features for 'Weekly_Sales' ---
print("⏳ Creating rolling average features for 'Weekly_Sales'...")
if 'Weekly_Sales' not in df.columns:  # Check again, though handled above
    warnings.warn(
        "⚠️ 'Weekly_Sales' column not found. "
        "Rolling average features (Rolling_3_Week_Avg_Sales, Rolling_5_Week_Avg_Sales) cannot be calculated from source. "
        "These columns will be created and filled with 0.",
        UserWarning
    )
    df['Rolling_3_Week_Avg_Sales'] = 0
    df['Rolling_5_Week_Avg_Sales'] = 0
else:
    # shift(1) ensures rolling averages use past data, not including the current week's sales
    df['Rolling_3_Week_Avg_Sales'] = df.groupby(['Store', 'Dept'])[
        'Weekly_Sales'].shift(1).rolling(window=3, min_periods=1).mean()
    df['Rolling_5_Week_Avg_Sales'] = df.groupby(['Store', 'Dept'])[
        'Weekly_Sales'].shift(1).rolling(window=5, min_periods=1).mean()
    print("✅ Rolling average features for 'Weekly_Sales' created.")

# --- 5. Encode Categorical Features ('Type' to 'Store_Type_Code') ---
print("⏳ Encoding 'Type' into 'Store_Type_Code'...")
if 'Type' in df.columns:
    try:
        # Create a mapping for 'Type' to numerical codes
        unique_types = df['Type'].astype('category').cat.categories
        type_mapping = {type_val: code for code,
                        type_val in enumerate(unique_types)}

        df[EXPECTED_TYPE_FEATURE] = df['Type'].map(type_mapping)

        if df[EXPECTED_TYPE_FEATURE].isnull().any():
            warnings.warn(
                f"⚠️ Some values in 'Type' were not in the learned mapping. Filling NaNs in '{EXPECTED_TYPE_FEATURE}' with -1.", UserWarning)
            # Use -1 for unmapped/NaN types
            df[EXPECTED_TYPE_FEATURE].fillna(-1, inplace=True)

        print(
            f"✅ 'Type' column encoded into '{EXPECTED_TYPE_FEATURE}'. Mapping: {type_mapping}")
    except Exception as e:
        warnings.warn(
            f"⚠️ Could not encode 'Type' column due to error: {e}. Defaulting '{EXPECTED_TYPE_FEATURE}' to 0.", UserWarning)
        df[EXPECTED_TYPE_FEATURE] = 0
else:
    warnings.warn(
        f"⚠️ 'Type' column not found in input data. "
        f"'{EXPECTED_TYPE_FEATURE}' will be created and filled with 0. This might impact model accuracy if 'Type' is an important feature.",
        UserWarning
    )
    df[EXPECTED_TYPE_FEATURE] = 0


# --- 6. Fill NaNs from Lags/Rolling Windows ---
# These NaNs are expected at the beginning of each group's time series
print("⏳ Filling NaNs generated from lag/rolling features with 0...")
for col in EXPECTED_LAG_ROLL_FEATURES:
    if col in df.columns:
        df[col].fillna(0, inplace=True)
    else:
        # This case should ideally be covered by sections 3 & 4 creating these columns with 0 if Weekly_Sales was missing
        warnings.warn(
            f"Expected lag/roll feature '{col}' was not found. Creating and filling with 0.", UserWarning)
        df[col] = 0
print("✅ NaNs from lag/rolling features filled.")


# --- 7. Drop Original Columns No Longer Needed ---
columns_to_drop = []
if 'Type' in df.columns and EXPECTED_TYPE_FEATURE in df.columns:
    columns_to_drop.append('Type')

# 'Date' is used to create other features. Model_training.py will select features, so 'Date' can remain for now
# if 'Date' in df.columns:
#     columns_to_drop.append('Date') # Uncomment if 'Date' must be removed from this file

if columns_to_drop:
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    print(f"✅ Dropped original columns: {columns_to_drop}")


# --- 8. Final Check for Expected Columns ---
# This is a safeguard. model_training.py has its own list of features.
# This check ensures all *engineered* features are present.
all_engineered_features = EXPECTED_LAG_ROLL_FEATURES + \
    [EXPECTED_TYPE_FEATURE] + ['DayOfWeek', 'Month', 'Year', 'WeekOfYear']
missing_engineered_features = [
    f for f in all_engineered_features if f not in df.columns]
if missing_engineered_features:
    warnings.warn(
        f"Critical Warning: The following engineered features are still missing: {missing_engineered_features}. This will likely cause errors in model training.", UserWarning)

print(f"📊 Final feature engineered dataset shape: {df.shape}")
print(f"📋 Final columns in feature_engineered_data: {list(df.columns)}")

# --- 9. Save Feature-Engineered Data ---
print(
    f"💾 Saving feature-engineered data to '{FEATURE_ENGINEERED_OUTPUT_FILE}'...")
try:
    df.to_csv(FEATURE_ENGINEERED_OUTPUT_FILE, index=False)
    print(
        f"✅ Feature-engineered data saved successfully to '{FEATURE_ENGINEERED_OUTPUT_FILE}'.")
except Exception as e:
    print(f"❌ Error saving feature-engineered data: {e}")

print("\n🎉 Feature engineering script finished.")
