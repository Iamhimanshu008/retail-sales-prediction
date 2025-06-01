# predict_uploaded_file.py
import pandas as pd
import joblib
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = "model/random_forest_model.pkl"

# INPUT_FILE_PATH: User should specify this or script could take it as an argument.
# For this example, it defaults to the feature_engineered_data.csv which would predict on training/test data.
# For actual new predictions, this path should point to a NEW file with the same feature structure.
DEFAULT_INPUT_FILE_PATH = os.path.join("output", "feature_engineered_data.csv")
# Example for a new file: INPUT_FILE_PATH = "data/new_unseen_features.csv" (must be feature engineered)

OUTPUT_PREDICTIONS_DIR = "output"  # Directory for predictions
# Template for output filename
OUTPUT_FILE_NAME_TEMPLATE = "predictions_on_{filename}.csv"

os.makedirs(OUTPUT_PREDICTIONS_DIR, exist_ok=True)


def predict_from_file(input_file_path: str):
    """
    Loads data from input_file_path, makes predictions using the trained model,
    and saves the predictions.
    """
    print(f"\n🚀 Starting prediction for file: {input_file_path}")

    # --- Load Trained Model ---
    print(f"⏳ Loading trained model from '{MODEL_PATH}'...")
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print(
            f"❌ Error: Model file '{MODEL_PATH}' not found. Please train the model first using model_training.py.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Get required features from the model
    if hasattr(model, 'feature_names_in_'):
        required_features = list(model.feature_names_in_)
        print(
            f"ℹ️ Model expects the following {len(required_features)} features: {required_features}")
    else:
        print("❌ Error: Model does not have 'feature_names_in_' attribute. Cannot determine required features.")
        print("   This might be due to an older scikit-learn version or an issue with model saving/loading.")
        return

    # --- Load and Prepare Input Data ---
    print(f"⏳ Loading input data from '{input_file_path}'...")
    if not os.path.exists(input_file_path):
        print(f"❌ Error: Input file '{input_file_path}' not found.")
        print("   Please ensure the file exists or update the input file path.")
        if input_file_path == DEFAULT_INPUT_FILE_PATH:
            print("   If you intended to use the feature-engineered file, please run feature_engineering.py first.")
        return

    try:
        # Try parsing 'Date' if it exists, but don't fail if it doesn't (e.g., already feature engineered)
        try:
            df_input = pd.read_csv(input_file_path, parse_dates=['Date'])
        except (ValueError, KeyError):  # KeyError if 'Date' not in CSV, ValueError if unparseable
            df_input = pd.read_csv(input_file_path)

        print(
            f"✅ Input data loaded successfully. Shape: {df_input.shape}, Columns: {list(df_input.columns)}")
    except Exception as e:
        print(
            f"❌ Error loading or parsing input data from '{input_file_path}': {e}")
        return

    # Keep a copy of the original data for output, including any extra columns
    df_for_output = df_input.copy()

    # Drop target column 'Weekly_Sales' if present in the input file (we are predicting it)
    if 'Weekly_Sales' in df_input.columns:
        print("ℹ️ 'Weekly_Sales' (target column) found in input data. It will be kept for comparison if available, but not used for prediction.")
        # We don't drop it from df_input yet, but will select only required_features for X_predict

    # --- Feature Alignment and Missing Value Handling for Prediction ---
    X_predict = pd.DataFrame()  # Initialize an empty DataFrame for features

    # Check for features required by the model but missing in the input data
    missing_from_input = list(set(required_features) - set(df_input.columns))
    if missing_from_input:
        print(
            f"⚠️ Warning: The input data is missing the following {len(missing_from_input)} features required by the model: {missing_from_input}")
        print("   These missing features will be filled with 0 by default. This might lead to inaccurate predictions if these features are important.")
        for col in missing_from_input:
            df_input[col] = 0  # Add missing columns and fill with 0

    # Select only the required features and ensure they are in the correct order
    try:
        X_predict = df_input[required_features]
        print(
            f"✅ Features aligned. Shape of X_predict for model: {X_predict.shape}")
    except KeyError as e:
        print(
            f"❌ Error: Could not align features for prediction. A required feature might be missing or there's a name mismatch. Missing key: {e}")
        print(
            f"   Available columns after potential fill: {list(df_input.columns)}")
        print(f"   Model requires: {required_features}")
        return

    # Handle any remaining NaNs in the final feature set (X_predict)
    # This is a final safety net. Ideally, all NaNs should be handled by upstream preprocessing or specific imputation.
    if X_predict.isnull().sum().any().any():  # Check if any column has any NaN
        print("⚠️ Warning: NaNs found in the final feature set for prediction. Filling them with 0.")
        X_predict = X_predict.fillna(0)

    # --- Make Predictions ---
    print("\n⏳ Predicting Weekly Sales...")
    try:
        predictions = model.predict(X_predict)
        print("✅ Predictions generated successfully.")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return

    # --- Save Results ---
    # Add predictions to the output DataFrame (which has original columns + potentially 'Weekly_Sales')
    df_for_output['Predicted_Weekly_Sales'] = predictions

    # Construct output file path
    base_filename = os.path.basename(input_file_path)
    output_filename = OUTPUT_FILE_NAME_TEMPLATE.format(
        filename=os.path.splitext(base_filename)[0])
    full_output_path = os.path.join(OUTPUT_PREDICTIONS_DIR, output_filename)

    print(f"\n💾 Saving predictions to '{full_output_path}'...")
    try:
        # Reorder columns to show prediction-related columns first for clarity
        cols_to_show_first = ['Predicted_Weekly_Sales']
        if 'Weekly_Sales' in df_for_output.columns:  # If actuals are available
            cols_to_show_first.append('Weekly_Sales')

        other_cols = [
            col for col in df_for_output.columns if col not in cols_to_show_first]
        df_for_output = df_for_output[cols_to_show_first + other_cols]

        df_for_output.to_csv(full_output_path, index=False)
        print(f"✅ Predictions saved successfully to '{full_output_path}'.")

        print("\n📄 Preview of predictions (first 5 rows):")
        print(df_for_output.head())
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")

    print(f"\n🎉 Prediction for file '{input_file_path}' finished.")


if __name__ == "__main__":
    # Determine input file: Use command-line argument if provided, else default.
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if not os.path.isfile(input_path):
            print(f"Error: Provided input file '{input_path}' does not exist.")
            print(f"Using default input file: '{DEFAULT_INPUT_FILE_PATH}'")
            input_path = DEFAULT_INPUT_FILE_PATH
    else:
        print(f"No input file path provided as command-line argument.")
        print(f"Using default input file: '{DEFAULT_INPUT_FILE_PATH}'")
        input_path = DEFAULT_INPUT_FILE_PATH
        if not os.path.exists(input_path):
            print(
                f"Warning: Default input file '{input_path}' not found. Prediction may fail.")
            print(f"You can run the script with a file path: python predict_uploaded_file.py /path/to/your/feature_engineered_file.csv")

    predict_from_file(input_path)
