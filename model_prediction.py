import pandas as pd
import joblib
import os
import warnings

# --- Configuration ---
FEATURE_ENGINEERED_FILE_PATH = 'output/feature_engineered_data.csv'
MODEL_PATH = 'model/random_forest_model.pkl'
OUTPUT_DIR = 'output'
PREDICTED_SALES_FILE_PATH = os.path.join(
    OUTPUT_DIR, 'batch_predicted_sales.csv')  # Renamed for clarity


def main():
    print("🚀 Starting batch prediction script...")

    # --- 1. Load Feature Engineered Data ---
    print(
        f"⏳ Loading feature-engineered data from '{FEATURE_ENGINEERED_FILE_PATH}'...")
    if not os.path.exists(FEATURE_ENGINEERED_FILE_PATH):
        print(
            f"❌ Error: Feature engineered data file '{FEATURE_ENGINEERED_FILE_PATH}' not found.")
        print("   Please run feature_engineering.py first to generate this file.")
        return
    try:
        df_predict = pd.read_csv(FEATURE_ENGINEERED_FILE_PATH)
        print(f"✅ Data loaded successfully. Shape: {df_predict.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # --- 2. Load Trained Model ---
    print(f"⏳ Loading trained model from '{MODEL_PATH}'...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
        print("   Please train the model first using model_training.py.")
        return
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- 3. Prepare Data for Prediction (Align Features) ---
    if not hasattr(model, 'feature_names_in_'):
        print("❌ Error: Model does not have 'feature_names_in_' attribute. Cannot reliably align features.")
        print("   This might be due to an older scikit-learn version or an issue with model training/saving.")
        return

    required_features = list(model.feature_names_in_)
    print(
        f"ℹ️ Model was trained with the following {len(required_features)} features: {required_features}")

    X_to_predict = pd.DataFrame()
    missing_cols = []

    for feature in required_features:
        if feature in df_predict.columns:
            X_to_predict[feature] = df_predict[feature]
        else:
            missing_cols.append(feature)
            X_to_predict[feature] = 0  # Default missing features to 0

    if missing_cols:
        warnings.warn(
            f"⚠️ The following required features were missing from '{FEATURE_ENGINEERED_FILE_PATH}' and were defaulted to 0: {missing_cols}. "
            "This can significantly impact prediction accuracy. Ensure 'feature_engineering.py' generates all necessary features.",
            UserWarning
        )

    # Handle any NaNs that might still be in the selected features (e.g., if 0 wasn't appropriate or some features were not defaulted by feature_engineering.py)
    if X_to_predict.isnull().sum().any():
        warnings.warn(
            "⚠️ NaNs found in features selected for prediction. Filling with 0. Review data pipeline.", UserWarning)
        X_to_predict.fillna(0, inplace=True)

    print(
        f"✅ Features prepared for prediction. Shape of X_to_predict: {X_to_predict.shape}")

    # --- 4. Make Predictions ---
    print("⏳ Predicting Weekly Sales...")
    try:
        predictions = model.predict(X_to_predict)
        df_predict['Predicted_Weekly_Sales'] = predictions
        print("✅ Predictions generated successfully.")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        # Potentially print X_to_predict.info() or X_to_predict.head() for debugging
        return

    # --- 5. Save Predictions ---
    # The df_predict DataFrame now contains original columns from feature_engineered_data.csv plus 'Predicted_Weekly_Sales'
    try:
        # Reorder to bring predicted sales to the front
        cols = ['Predicted_Weekly_Sales'] + \
            [col for col in df_predict.columns if col != 'Predicted_Weekly_Sales']
        if 'Weekly_Sales' in cols:  # If actuals were present, put them next to predictions
            cols.pop(cols.index('Weekly_Sales'))
            cols.insert(1, 'Weekly_Sales')

        df_output = df_predict[cols]

        df_output.to_csv(PREDICTED_SALES_FILE_PATH, index=False)
        print(f"✅ Predictions saved to '{PREDICTED_SALES_FILE_PATH}'")
        print("\n📄 Preview of predictions (first 5 rows):")
        print(df_output.head())
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")

    print("\n🎉 Batch prediction script finished.")


if __name__ == "__main__":
    main()
