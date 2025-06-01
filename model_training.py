import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import numpy as np
import warnings

# --- Configuration ---
FEATURE_ENGINEERED_FILE = 'output/feature_engineered_data.csv'
MODEL_DIR = 'model'
MODEL_FILE = os.path.join(MODEL_DIR, 'random_forest_model.pkl')

# Features and target columns
# THIS LIST MUST EXACTLY MATCH THE COLUMNS INTENDED FOR MODELING
# AND MUST BE PRESENT IN THE FEATURE_ENGINEERED_FILE
FEATURES = [
    'Store', 'Dept', 'IsHoliday',
    'Temperature', 'Fuel_Price',
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
    'CPI', 'Unemployment', 'Size',
    'Month', 'Year', 'WeekOfYear',
    'DayOfWeek',  # Integer day of week (Monday=0, Sunday=6)
    'Lag_1_Week_Sales', 'Lag_2_Week_Sales',
    'Rolling_3_Week_Avg_Sales', 'Rolling_5_Week_Avg_Sales',
    'Store_Type_Code'  # Encoded 'Type' column
]
TARGET = 'Weekly_Sales'


def load_data(file_path: str) -> pd.DataFrame:
    """Load feature engineered data from CSV."""
    print(f"⏳ Loading feature-engineered data from '{file_path}'...")
    if not os.path.exists(file_path):
        print(
            f"❌ Error: Feature engineered data file '{file_path}' not found.")
        raise FileNotFoundError(
            f"Please run feature_engineering.py first to generate '{file_path}'.")

    try:
        df = pd.read_csv(file_path)
        print(
            f"✅ Loaded data successfully. Shape: {df.shape}, Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Error loading data from '{file_path}': {e}")
        raise


def validate_columns(df: pd.DataFrame, features: list, target: str):
    """Ensure all features and target columns exist in the dataframe."""
    print("🕵️ Validating presence of required columns...")
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        error_msg = (f"❌ Critical Error: Missing feature columns in the loaded data: {missing_features}. "
                     f"Ensure 'feature_engineering.py' produces these columns or update the FEATURES list.")
        print(error_msg)
        raise ValueError(error_msg)

    if target not in df.columns:
        error_msg = f"❌ Critical Error: Target column '{target}' missing in the loaded data."
        print(error_msg)
        raise ValueError(error_msg)

    print("✅ All required feature and target columns are present in the data.")


def prepare_features_target(df: pd.DataFrame, features: list, target: str):
    """Select features and target, and handle missing values in them."""
    print("⚙️ Preparing features (X) and target (y)...")
    X = df[features].copy()
    y = df[target].copy()

    print("✨ Imputing missing values in features (X)...")
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(X[col]):
                fill_value = X[col].median()
                X[col].fillna(fill_value, inplace=True)
                print(
                    f"  -> Imputed NaNs in numeric feature '{col}' with median ({fill_value:.2f}).")
            else:  # Assuming non-numeric are categorical - though all FEATURES here should be numeric by now
                fill_value = X[col].mode()[0]
                X[col].fillna(fill_value, inplace=True)
                print(
                    f"  -> Imputed NaNs in categorical feature '{col}' with mode ({fill_value}).")

    if y.isnull().sum() > 0:
        print("✨ Imputing missing values in target (y)...")
        # Or 0, or drop rows: y.dropna(inplace=True); X = X.loc[y.index]
        fill_value = y.median()
        y.fillna(fill_value, inplace=True)
        print(
            f"  -> Imputed NaNs in target '{target}' with median ({fill_value:.2f}).")

    print("✅ Features and target prepared.")
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    print(
        f"🔪 Splitting data into training and testing sets (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    print(
        f"  Train set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    print("✅ Data splitting complete.")
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Train RandomForestRegressor model."""
    print("🏋️ Training RandomForestRegressor model...")
    print("  Model Hyperparameters: n_estimators=100, max_depth=15, max_features='sqrt', random_state=42, n_jobs=-1")
    print("  This might take some time depending on data size and system resources...")

    # For datasets that are extremely large, consider reducing n_estimators or max_depth,
    # or using a subset of data for faster iterations.
    # If performance is still an issue, LightGBM or XGBoost are faster alternatives.
    model = RandomForestRegressor(
        n_estimators=100,     # Number of trees in the forest
        max_depth=15,         # Maximum depth of the tree
        min_samples_split=2,  # Minimum number of samples required to split an internal node
        min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
        # Number of features to consider when looking for the best split (sqrt(n_features))
        max_features='sqrt',
        random_state=42,      # For reproducibility
        n_jobs=-1             # Use all available cores
    )
    model.fit(X_train, y_train)
    print("✅ Model training completed.")
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance on train and test sets."""
    print("\n📈 Evaluating model performance...")

    sets = {
        "Training Set": (X_train, y_train),
        "Test Set": (X_test, y_test)
    }

    for name, (X_data, y_true) in sets.items():
        y_pred = model.predict(X_data)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print(f"\n  Metrics for {name}:")
        print(f"    R² Score: {r2:.4f}")
        print(f"    Mean Absolute Error (MAE): {mae:.2f}")
        print(f"    Root Mean Squared Error (RMSE): {rmse:.2f}")
    print("✅ Model evaluation complete.")


def save_model(model, model_dir: str, model_filename: str):
    """Save trained model to disk."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"📁 Created model directory: '{model_dir}'")

    model_path = os.path.join(model_dir, model_filename)
    try:
        joblib.dump(model, model_path)
        print(f"💾 Model saved successfully at '{model_path}'")
    except Exception as e:
        print(f"❌ Error saving model to '{model_path}': {e}")
        raise


def main():
    print("🚀 Starting model training pipeline...")
    try:
        df = load_data(FEATURE_ENGINEERED_FILE)
        validate_columns(df, FEATURES, TARGET)
        X, y = prepare_features_target(df, FEATURES, TARGET)
        X_train, X_test, y_train, y_test = split_data(X, y)

        model = train_model(X_train, y_train)

        # Store feature names in the model object (Scikit-learn >= 0.24 does this automatically if trained on DataFrame)
        # This is good for consistency, but usually handled by sklearn if input is a DataFrame.
        # model.feature_names_in_ = list(X_train.columns) # Explicitly setting, though often not needed with modern sklearn

        evaluate_model(model, X_train, y_train, X_test, y_test)
        save_model(model, MODEL_DIR, os.path.basename(MODEL_FILE))

        print("\n🎉 Model training pipeline completed successfully!")

    except FileNotFoundError:
        # Error already printed by load_data
        print("Please ensure the prerequisite scripts have run and files exist.")
    except ValueError as ve:
        # Error already printed by validate_columns or prepare_features_target
        print(f"A ValueError occurred: {ve}")
    except Exception as e:
        print(
            f"❌ An unexpected error occurred during the training pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
