# retail_analytics_project/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
CLEANED_DATA_FILE = 'output/cleaned_data.csv'
OUTPUT_DIR = 'output/eda_plots'  # Directory to save plots
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load cleaned data
print(f"⏳ Loading cleaned data from '{CLEANED_DATA_FILE}' for EDA...")
if not os.path.exists(CLEANED_DATA_FILE):
    print(f"❌ Error: Cleaned data file '{CLEANED_DATA_FILE}' not found.")
    print("Please run data_preprocessing.py script first.")
    exit()

try:
    df = pd.read_csv(CLEANED_DATA_FILE, parse_dates=['Date'])
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading cleaned data: {e}")
    exit()

print("📊 Generating EDA plots...")

# 1. Total Weekly Sales over Time
plt.figure(figsize=(14, 7))
df.groupby('Date')['Weekly_Sales'].sum().plot(
    kind='line', marker='.', linestyle='-')
plt.title('Total Weekly Sales Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Weekly Sales', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'total_weekly_sales_over_time.png'))
print(f"✅ Plot saved: total_weekly_sales_over_time.png")
# plt.show() # Uncomment if you want to display interactively

# 2. Store-wise Total Sales (Top 20 Stores)
store_sales = df.groupby(
    'Store')['Weekly_Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(14, 7))
store_sales.head(20).plot(kind='bar', color=sns.color_palette("viridis", 20))
plt.title('Top 20 Stores by Total Sales', fontsize=16)
plt.xlabel('Store ID', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'store_wise_total_sales_top20.png'))
print(f"✅ Plot saved: store_wise_total_sales_top20.png")
# plt.show()

# 3. Sales Distribution (Histogram)
plt.figure(figsize=(12, 6))
sns.histplot(df['Weekly_Sales'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Weekly Sales per Department per Week', fontsize=16)
plt.xlabel('Weekly Sales', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'weekly_sales_distribution.png'))
print(f"✅ Plot saved: weekly_sales_distribution.png")
# plt.show()

# 4. Sales Distribution by Store Type
if 'Type' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Type', y='Weekly_Sales', data=df, palette='pastel')
    plt.title('Weekly Sales Distribution by Store Type', fontsize=16)
    plt.xlabel('Store Type', fontsize=12)
    plt.ylabel('Weekly Sales', fontsize=12)
    # Zoom in, excluding outliers
    plt.ylim(df['Weekly_Sales'].quantile(0.01),
             df['Weekly_Sales'].quantile(0.99))
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sales_by_store_type.png'))
    print(f"✅ Plot saved: sales_by_store_type.png")
    # plt.show()
else:
    print("ℹ️ 'Type' column not found, skipping 'Sales Distribution by Store Type' plot.")


# 5. Correlation Heatmap of Numerical Features
numerical_features = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size',
                      'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'IsHoliday']
# Filter out features not present in df to avoid KeyError
numerical_features_present = [
    col for col in numerical_features if col in df.columns]

if numerical_features_present:
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numerical_features_present].corr()
    sns.heatmap(correlation_matrix, annot=True,
                cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
    print(f"✅ Plot saved: correlation_heatmap.png")
    # plt.show()
else:
    print("ℹ️ No numerical features found for correlation heatmap.")


# 6. Impact of Holidays on Sales
if 'IsHoliday' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='IsHoliday', y='Weekly_Sales', data=df, palette='coolwarm')
    plt.title('Impact of Holidays on Weekly Sales', fontsize=16)
    plt.xlabel('Is Holiday (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel('Weekly Sales', fontsize=12)
    plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
    plt.ylim(df['Weekly_Sales'].quantile(0.01),
             df['Weekly_Sales'].quantile(0.99))  # Zoom in
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'holiday_impact_on_sales.png'))
    print(f"✅ Plot saved: holiday_impact_on_sales.png")
    # plt.show()
else:
    print("ℹ️ 'IsHoliday' column not found, skipping 'Impact of Holidays on Sales' plot.")

print("\n🎉 EDA script finished. Plots saved in 'output/eda_plots/' directory.")
