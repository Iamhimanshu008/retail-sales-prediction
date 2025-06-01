# feature_importance.py
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import seaborn as sns

# --- Configuration ---
MODEL_FILE = 'model/random_forest_model.pkl'
OUTPUT_DIR = 'output'  # Main output directory
FEATURE_IMPORTANCE_PLOT_FILE = os.path.join(
    # Changed filename slightly for clarity
    OUTPUT_DIR, 'feature_importance_plot.png')

# Ensure output directory exists (though 'output' should exist from other scripts)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load Model ---
print("🚀 Starting feature importance script...")
print(f"⏳ Loading trained model from '{MODEL_FILE}'...")
if not os.path.exists(MODEL_FILE):
    print(f"❌ Error: Model file '{MODEL_FILE}' not found.")
    print("Please train the model first by running model_training.py.")
    exit()
try:
    model = joblib.load(MODEL_FILE)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# --- 2. Get Feature Importances ---
print("📊 Extracting feature importances...")
if not hasattr(model, 'feature_importances_'):
    print("❌ Error: The loaded model does not have 'feature_importances_' attribute.")
    print("Ensure it's a scikit-learn tree-based model (e.g., RandomForest, GradientBoosting) and has been fitted.")
    exit()

importances = model.feature_importances_

# Get feature names from the model directly (most reliable way)
if hasattr(model, 'feature_names_in_'):
    features = list(model.feature_names_in_)
    print(
        f"✅ Features obtained from model.feature_names_in_ ({len(features)} features).")
else:
    # Fallback for older scikit-learn versions or models not trained with DataFrames directly
    # This part is less ideal and assumes the model was trained with a specific list of features
    # that needs to be manually provided or inferred, which is error-prone.
    # For this project, model_training.py uses DataFrames, so feature_names_in_ should be present.
    print("❌ Error: Model does not have 'feature_names_in_' attribute.")
    print("This typically means the scikit-learn version is older (<0.24) or the model wasn't fit with a DataFrame with feature names.")
    print("Cannot reliably get feature names. Ensure scikit-learn is up-to-date and model was trained with named features.")
    # As a last resort, if you KNOW the features, you could define them here:
    # features = ['Feature1', 'Feature2', ...] # But this is not recommended
    exit()

# Consistency check
if len(features) != len(importances):
    print(f"❌ Mismatch Error: Number of features from model ({len(features)}) "
          f"does not match number of importances ({len(importances)}). This should not happen.")
    # This might occur if the 'features' list above was manually defined and incorrect.
    exit()

# --- 3. Create DataFrame for Plotting ---
feature_importance_df = pd.DataFrame(
    {'Feature': features, 'Importance': importances})
# Sort by importance in descending order
feature_importance_df = feature_importance_df.sort_values(
    by='Importance', ascending=False).reset_index(drop=True)

print("\nTop 10 Feature Importances:")
print(feature_importance_df.head(10))

# --- 4. Plot Feature Importances ---
print("\n📈 Plotting feature importances...")
plt.figure(figsize=(12, 10))  # Adjusted for potentially many features
# Plot top 25 or all if fewer
num_features_to_plot = min(len(feature_importance_df), 25)

sns.barplot(x='Importance', y='Feature',
            data=feature_importance_df.head(num_features_to_plot),
            palette='viridis_r')  # Using a reversed viridis palette

plt.title(
    f'Top {num_features_to_plot} Feature Importances from RandomForestRegressor', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.gca().invert_yaxis()  # Display most important at the top
plt.tight_layout()  # Adjust layout to prevent labels from overlapping

# --- 5. Save Plot ---
try:
    plt.savefig(FEATURE_IMPORTANCE_PLOT_FILE)
    print(
        f"✅ Feature importance plot saved to '{FEATURE_IMPORTANCE_PLOT_FILE}'")
    # plt.show() # Uncomment to display plot if running interactively
except Exception as e:
    print(f"❌ Error saving feature importance plot: {e}")

print("\n🎉 Feature importance script finished.")
