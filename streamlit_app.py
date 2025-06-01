# Streamlit App for Retail Sales Prediction
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go  # For more control if needed
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Retail Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🛒 Retail Sales Prediction Dashboard")

# --- Paths ---
# Assuming streamlit_app.py is at the root of the project or paths are relative from where it's run
MODEL_PATH = "model/random_forest_model.pkl"
CLEANED_DATA_PATH = "output/cleaned_data.csv"  # For EDA
FEATURE_ENGINEERED_DATA_PATH = "output/feature_engineered_data.csv"  # For Prediction Tab
FEEDBACK_DIR = "output/feedback"
FEEDBACK_PATH = os.path.join(FEEDBACK_DIR, "feedback_log.txt")

# Ensure output directories exist
os.makedirs("output", exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs("model", exist_ok=True)  # Ensure model directory exists


# --- Data and Model Loading ---
@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            if not hasattr(model, 'feature_names_in_'):
                st.error(
                    f"Model at '{model_path}' loaded, but missing 'feature_names_in_'. May be incompatible.", icon="⚠️")
            return model
        except Exception as e:
            st.error(f"Error loading model from '{model_path}': {e}", icon="❌")
            return None
    return None


@st.cache_data
def load_dataframe(file_path, parse_dates=None):
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, parse_dates=parse_dates)
        except Exception as e:
            st.error(f"Error loading data from '{file_path}': {e}", icon="❌")
            return None
    return None


# Load resources
model = load_model(MODEL_PATH)
df_cleaned_eda = load_dataframe(CLEANED_DATA_PATH, parse_dates=['Date'])
df_feature_engineered_full = load_dataframe(FEATURE_ENGINEERED_DATA_PATH, parse_dates=[
                                            'Date'])  # Keep Date if present for filtering

# --- Sidebar ---
st.sidebar.title("⚙️ Controls & Info")
st.sidebar.markdown("Navigate through different sections and apply filters.")

page_options = ["🏠 Home", "🔮 Predict Sales",
                "📊 Exploratory Data Analysis", "📝 Feedback"]
selected_page = st.sidebar.radio("Go to Page:", page_options)

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Data & Model Status")
if df_cleaned_eda is not None:
    st.sidebar.success(
        f"Cleaned EDA data loaded ({df_cleaned_eda.shape[0]} rows).")
else:
    st.sidebar.warning(f"Cleaned EDA data ('{CLEANED_DATA_PATH}') not found.")

if df_feature_engineered_full is not None:
    st.sidebar.success(
        f"Feature-engineered data loaded ({df_feature_engineered_full.shape[0]} rows).")
else:
    st.sidebar.warning(
        f"Feature-engineered data ('{FEATURE_ENGINEERED_DATA_PATH}') not found.")

if model:
    st.sidebar.success("✅ Prediction Model Loaded")
    if hasattr(model, 'feature_names_in_'):
        st.sidebar.markdown(
            f"Model expects **{len(model.feature_names_in_)} features**.")
        with st.sidebar.expander("Show Expected Model Features"):
            st.json(list(model.feature_names_in_))
else:
    st.sidebar.error("❌ Prediction Model Not Found or Failed to Load!")
    st.sidebar.markdown(f"Ensure model is at: `{MODEL_PATH}`")

if os.path.exists(FEATURE_ENGINEERED_DATA_PATH):
    try:
        with open(FEATURE_ENGINEERED_DATA_PATH, "rb") as fp:
            st.sidebar.download_button(
                label="📥 Download Example Feature-Engineered CSV",
                data=fp,
                file_name="example_feature_engineered_data.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.sidebar.error(f"Could not provide example CSV: {e}")


# --- Helper function for prediction and display ---
def display_predictions_and_metrics(df_to_predict_on, model_obj, title_prefix=""):
    if df_to_predict_on.empty:
        st.info(
            f"{title_prefix}No data to predict on after filtering. Adjust filters or upload data.")
        return

    st.markdown(f"#### {title_prefix}Prediction Results")

    # Prepare data for prediction
    df_for_prediction_features = df_to_predict_on.copy()
    y_true_for_metrics = None

    if 'Weekly_Sales' in df_for_prediction_features.columns:
        y_true_for_metrics = df_for_prediction_features['Weekly_Sales'].copy()
        # Do not drop 'Weekly_Sales' if it's somehow in model.feature_names_in_
        # This check is more for safety, it should not be a feature.
        if 'Weekly_Sales' in model_obj.feature_names_in_:
            st.warning(
                "'Weekly_Sales' is in model.feature_names_in_! This is unusual. It will be used as a feature.", icon="⚠️")

    required_cols = model_obj.feature_names_in_
    X_predict = pd.DataFrame()
    missing_model_cols = []

    for col in required_cols:
        if col in df_for_prediction_features.columns:
            X_predict[col] = df_for_prediction_features[col]
        else:
            X_predict[col] = 0  # Fill missing required columns with 0
            missing_model_cols.append(col)

    if missing_model_cols:
        st.warning(
            f"The following model features were missing and defaulted to 0: **{', '.join(missing_model_cols)}**. This may affect accuracy.", icon="⚠️")

    if X_predict.isnull().sum().any():
        st.warning(
            "NaNs found in feature columns for prediction. Filling with 0.", icon="⚠️")
        X_predict = X_predict.fillna(0)

    with st.spinner("🧠 Generating predictions..."):
        predictions_array = model_obj.predict(X_predict[required_cols])
    st.success("✅ Predictions generated!")

    # Use original df for display context
    result_df_display = df_to_predict_on.copy()
    result_df_display["Predicted_Sales"] = predictions_array

    display_cols_order = ["Predicted_Sales"]
    if y_true_for_metrics is not None:
        # Actual sales if available
        display_cols_order.insert(0, "Weekly_Sales")

    id_cols_order = ['Store', 'Dept', 'Date']  # Key identifiers
    for id_col in id_cols_order:
        if id_col in result_df_display.columns and id_col not in display_cols_order:
            display_cols_order.append(id_col)

    other_display_cols = [
        col for col in result_df_display.columns if col not in display_cols_order and col != "Predicted_Sales"]
    st.dataframe(
        result_df_display[display_cols_order + other_display_cols].head(10))

    # Performance Metrics
    if y_true_for_metrics is not None and not y_true_for_metrics.empty:
        st.markdown("##### 📊 Performance Metrics")
        col1, col2, col3 = st.columns(3)
        r2 = r2_score(y_true_for_metrics, predictions_array)
        mae = mean_absolute_error(y_true_for_metrics, predictions_array)
        rmse = np.sqrt(mean_squared_error(
            y_true_for_metrics, predictions_array))

        col1.metric(
            "R² Score", f"{r2:.4f}", help="Coefficient of determination. Closer to 1 is better.")
        col2.metric("Mean Absolute Error (MAE)",
                    f"{mae:,.2f}", help="Average absolute difference.")
        col3.metric("Root Mean Squared Error (RMSE)",
                    f"{rmse:,.2f}", help="Penalizes large errors more.")

        st.markdown("##### 📈 Actual vs. Predicted Sales")
        fig_scatter = px.scatter(
            x=y_true_for_metrics, y=predictions_array,
            labels={'x': 'Actual Sales', 'y': 'Predicted Sales'},
            title="Actual vs. Predicted Sales Comparison",
            trendline="ols", trendline_color_override="red"
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("##### 🔍 Predicted Sales Distribution")
    fig_hist = px.histogram(
        result_df_display, x="Predicted_Sales", nbins=50,
        title="Distribution of Predicted Weekly Sales",
        labels={'Predicted_Sales': 'Predicted Sales Amount'}
    )
    fig_hist.update_layout(height=500)
    st.plotly_chart(fig_hist, use_container_width=True)


# --- Page Content ---

if selected_page == "🏠 Home":
    st.header("🏠 Welcome to the Retail Sales Predictor!")
    st.markdown("""
    This interactive dashboard leverages a machine learning model to predict weekly sales for retail stores and departments.

    **Explore the App:**
    - **🔮 Predict Sales:**
        - View predictions on pre-loaded, feature-engineered data.
        - Apply filters (Store, Department, Year, Month) to analyze specific segments.
        - Optionally, upload your own feature-engineered CSV file for custom predictions.
    - **📊 Exploratory Data Analysis (EDA):**
        - Discover insights from the cleaned dataset through various visualizations like sales trends, holiday impact, and distributions.
    - **📝 Feedback:**
        - Share your thoughts or report any issues to help us improve this tool.

    **Getting Started:**
    - Ensure the necessary data files (`cleaned_data.csv`, `feature_engineered_data.csv`) and the model (`random_forest_model.pkl`) are present in the `output` and `model` directories respectively.
    - The sidebar shows the status of these components.
    """)
    st.markdown("---")
    if model and hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
        st.subheader("⭐ Top Model Feature Importances")
        try:
            importances = model.feature_importances_
            features = model.feature_names_in_
            feature_importance_df = pd.DataFrame(
                {'Feature': features, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values(
                by='Importance', ascending=False).head(10)

            fig_importance = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h',
                                    title='Top 10 Most Important Features in the Model',
                                    labels={'Feature': 'Feature Name', 'Importance': 'Importance Score'})
            fig_importance.update_layout(
                yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
        except Exception as e:
            st.warning(
                f"Could not display feature importances: {e}", icon="⚠️")


elif selected_page == "🔮 Predict Sales":
    st.header("🔮 Predict Weekly Sales")

    if not model:
        st.error("❌ Prediction Model Not Loaded. Cannot make predictions.", icon="🚨")
        st.info(
            f"Ensure the model file is present at `{MODEL_PATH}` and the application is restarted if needed.")
    elif df_feature_engineered_full is None:
        st.error(
            f"❌ Feature-engineered data ('{FEATURE_ENGINEERED_DATA_PATH}') not found. Cannot show default predictions.", icon="🚨")
    else:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Filters for Default Predictions")

        df_filtered_for_prediction = df_feature_engineered_full.copy()

        # Filter by Store
        if 'Store' in df_filtered_for_prediction.columns:
            unique_stores = sorted(
                df_filtered_for_prediction['Store'].unique())
            selected_stores = st.sidebar.multiselect("Select Store(s):", unique_stores, default=unique_stores[:min(
                3, len(unique_stores))])  # Default to first 3 or all
            if selected_stores:
                df_filtered_for_prediction = df_filtered_for_prediction[df_filtered_for_prediction['Store'].isin(
                    selected_stores)]

        # Filter by Department
        if 'Dept' in df_filtered_for_prediction.columns:
            unique_depts = sorted(df_filtered_for_prediction['Dept'].unique())
            selected_depts = st.sidebar.multiselect(
                "Select Department(s):", unique_depts, default=[])
            if selected_depts:
                df_filtered_for_prediction = df_filtered_for_prediction[df_filtered_for_prediction['Dept'].isin(
                    selected_depts)]

        # Filter by Year and Month (assuming Year and Month columns exist from feature_engineering.py)
        if 'Year' in df_filtered_for_prediction.columns:
            unique_years = sorted(df_filtered_for_prediction['Year'].unique())
            selected_year = st.sidebar.selectbox(
                "Select Year:", ["All"] + unique_years, index=0)
            if selected_year != "All":
                df_filtered_for_prediction = df_filtered_for_prediction[
                    df_filtered_for_prediction['Year'] == selected_year]

        if 'Month' in df_filtered_for_prediction.columns:
            unique_months = sorted(
                df_filtered_for_prediction['Month'].unique())
            selected_month = st.sidebar.selectbox(
                "Select Month:", ["All"] + unique_months, index=0)
            if selected_month != "All":
                df_filtered_for_prediction = df_filtered_for_prediction[
                    df_filtered_for_prediction['Month'] == selected_month]

        st.markdown(
            "### Default Predictions (from loaded feature-engineered data)")
        st.info("Predictions below are based on the pre-loaded `feature_engineered_data.csv` and active filters from the sidebar.", icon="ℹ️")
        display_predictions_and_metrics(
            df_filtered_for_prediction, model, title_prefix="Default ")

        st.markdown("---")
        st.subheader("📤 Upload Your Own Feature-Engineered CSV for Prediction")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="CSV should contain all features the model was trained on. 'Weekly_Sales' (if present) will be used for metrics."
        )
        if uploaded_file:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                st.success(
                    f"✅ File '{uploaded_file.name}' uploaded. Shape: {df_uploaded.shape}")
                display_predictions_and_metrics(
                    df_uploaded, model, title_prefix="Uploaded File ")
            except Exception as e:
                st.error(f"❌ Error processing uploaded file: {e}", icon="🚨")


elif selected_page == "📊 Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis (from Cleaned Data)")
    st.markdown(
        f"Visualizations based on `cleaned_data.csv` (if available at `{CLEANED_DATA_PATH}`).")

    if df_cleaned_eda is not None:
        st.success("✅ Cleaned data for EDA loaded successfully.")

        # Plot 1: Sales Over Time
        st.subheader("🗓️ Total Weekly Sales Over Time")
        if 'Date' in df_cleaned_eda.columns and 'Weekly_Sales' in df_cleaned_eda.columns:
            try:
                sales_over_time = df_cleaned_eda.groupby(
                    'Date')['Weekly_Sales'].sum().reset_index()
                fig_time = px.line(
                    sales_over_time, x='Date', y='Weekly_Sales', title="Total Weekly Sales Trend")
                st.plotly_chart(fig_time, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot sales over time: {e}", icon="⚠️")
        else:
            st.warning(
                "'Date' or 'Weekly_Sales' columns missing for 'Sales Over Time' plot.", icon="⚠️")

        # Plot 2: Holiday vs Non-Holiday Sales
        st.subheader("⚖️ Sales: Holiday vs Non-Holiday")
        if 'IsHoliday' in df_cleaned_eda.columns and 'Weekly_Sales' in df_cleaned_eda.columns:
            try:
                fig_holiday = px.box(df_cleaned_eda, x='IsHoliday', y='Weekly_Sales',
                                     title="Weekly Sales: Holiday vs Non-Holiday",
                                     labels={
                                         'IsHoliday': 'Is Holiday? (0=No, 1=Yes)', 'Weekly_Sales': 'Weekly Sales'},
                                     points="outliers", color="IsHoliday",
                                     category_orders={"IsHoliday": [0, 1]})
                st.plotly_chart(fig_holiday, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot holiday sales: {e}", icon="⚠️")

        else:
            st.warning(
                "'IsHoliday' or 'Weekly_Sales' columns missing for 'Holiday Sales' plot.", icon="⚠️")

        # Plot 3: Sales by Store Type
        st.subheader("🏬 Sales by Store Type")
        if 'Type' in df_cleaned_eda.columns and 'Weekly_Sales' in df_cleaned_eda.columns:
            try:
                fig_type_sales = px.box(df_cleaned_eda, x='Type', y='Weekly_Sales',
                                        title="Weekly Sales by Store Type",
                                        labels={'Type': 'Store Type',
                                                'Weekly_Sales': 'Weekly Sales'},
                                        points="outliers", color="Type")
                st.plotly_chart(fig_type_sales, use_container_width=True)
            except Exception as e:
                st.warning(
                    f"Could not plot sales by store type: {e}", icon="⚠️")
        else:
            st.warning(
                "Required 'Type' or 'Weekly_Sales' columns missing for 'Sales by Store Type' plot.", icon="⚠️")

        # Plot 4: Top N Stores by Sales (Matplotlib/Seaborn example)
        st.subheader("🏆 Top Stores by Total Sales")
        if 'Store' in df_cleaned_eda.columns and 'Weekly_Sales' in df_cleaned_eda.columns:
            try:
                top_n_stores = st.slider(
                    "Select number of top stores to display:", 5, 20, 10, key="top_stores_slider_eda")
                store_total_sales = df_cleaned_eda.groupby('Store')['Weekly_Sales'].sum(
                ).nlargest(top_n_stores).sort_values(ascending=False)

                fig_top_stores, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=store_total_sales.values, y=store_total_sales.index.astype(
                    str), ax=ax, palette="viridis", orient='h')
                ax.set_title(f'Top {top_n_stores} Stores by Total Sales')
                ax.set_xlabel('Total Sales')
                ax.set_ylabel('Store ID')
                st.pyplot(fig_top_stores)
                # Important for Matplotlib in Streamlit
                plt.close(fig_top_stores)
            except Exception as e:
                st.warning(f"Could not plot top stores: {e}", icon="⚠️")
        else:
            st.warning(
                "Required 'Store' or 'Weekly_Sales' columns missing for 'Top Stores' plot.", icon="⚠️")

        # Plot 5: Top N Departments by Sales
        st.subheader("🛍️ Top Departments by Total Sales")
        if 'Dept' in df_cleaned_eda.columns and 'Weekly_Sales' in df_cleaned_eda.columns:
            try:
                top_n_depts = st.slider(
                    "Select number of top departments to display:", 5, 20, 10, key="top_depts_slider_eda")
                dept_total_sales = df_cleaned_eda.groupby('Dept')['Weekly_Sales'].sum(
                ).nlargest(top_n_depts).sort_values(ascending=False)

                fig_top_depts, ax_dept = plt.subplots(figsize=(10, 6))
                sns.barplot(x=dept_total_sales.values, y=dept_total_sales.index.astype(
                    str), ax=ax_dept, palette="mako", orient='h')
                ax_dept.set_title(
                    f'Top {top_n_depts} Departments by Total Sales')
                ax_dept.set_xlabel('Total Sales')
                ax_dept.set_ylabel('Department ID')
                st.pyplot(fig_top_depts)
                plt.close(fig_top_depts)
            except Exception as e:
                st.warning(f"Could not plot top departments: {e}", icon="⚠️")
        else:
            st.warning(
                "Required 'Dept' or 'Weekly_Sales' columns missing for 'Top Departments' plot.", icon="⚠️")

        # Plot 6: Correlation Heatmap
        st.subheader("🔗 Correlation Heatmap of Numerical Features")
        numerical_cols_for_corr = ['Weekly_Sales', 'Temperature',
                                   'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'IsHoliday']
        present_numerical_cols = [
            col for col in numerical_cols_for_corr if col in df_cleaned_eda.columns]
        if len(present_numerical_cols) > 1:
            try:
                corr_matrix = df_cleaned_eda[present_numerical_cols].corr()
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                            fmt=".2f", linewidths=.5, ax=ax_corr)
                ax_corr.set_title('Correlation Heatmap')
                st.pyplot(fig_corr)
                plt.close(fig_corr)
            except Exception as e:
                st.warning(
                    f"Could not plot correlation heatmap: {e}", icon="⚠️")
        else:
            st.warning(
                "Not enough numerical columns present for a meaningful correlation heatmap.", icon="⚠️")

    else:
        st.warning(
            f"⚠️ Cleaned data file (`{CLEANED_DATA_PATH}`) not found. EDA visualizations cannot be displayed.", icon="ℹ️")
        st.info("Please run `data_preprocessing.py` to generate the cleaned data file.")


elif selected_page == "📝 Feedback":
    st.header("📝 We Value Your Feedback!")
    st.markdown(
        "Share your thoughts, suggestions, or any issues. Your input helps us improve!")

    with st.form(key="feedback_form", clear_on_submit=True):
        user_name = st.text_input("Your Name (Optional)")
        feedback_type = st.selectbox("Feedback Type", [
                                     "Suggestion", "Bug Report", "Feature Request", "Compliment", "Other"])
        feedback_text = st.text_area(
            "Your Feedback*", height=150, placeholder="Enter your feedback here...")

        submit_button = st.form_submit_button(label="📤 Submit Feedback")

        if submit_button:
            if feedback_text.strip():
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"[{timestamp}] Name: {user_name if user_name else 'Anonymous'} | Type: {feedback_type} | Feedback: {feedback_text}\n---\n"

                try:
                    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
                        f.write(log_entry)
                    st.success(
                        "✅ Thank you for your feedback! It has been submitted.")
                except Exception as e:
                    st.error(
                        f"❌ Oops! Could not save your feedback: {e}", icon="🚨")
            else:
                st.error(
                    "⚠️ Feedback cannot be empty. Please write something.", icon="🚨")

    st.markdown("---")
    if os.path.exists(FEEDBACK_PATH):
        with st.expander("📜 View Submitted Feedback Log (for admin/dev review)"):
            try:
                with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                    feedback_content = f.read()
                st.text_area("Feedback Log", feedback_content, height=300,
                             disabled=True, help="This log is for review purposes.")
            except Exception as e:
                st.error(f"Could not read feedback log: {e}", icon="🚨")

# --- Footer (Optional) ---
st.sidebar.markdown("---")
st.sidebar.info("Retail Sales Predictor | Developed with Streamlit")
