# Retail Sales Prediction

This project aims to predict weekly retail sales for various stores and departments using machine learning. The pipeline includes data preprocessing, feature engineering, model training (Random Forest Regressor), and an interactive Streamlit dashboard for visualizing predictions and performing exploratory data analysis.


## Features

*   **Data Preprocessing:** Handles missing values, data type conversions, and merges multiple data sources.
*   **Feature Engineering:** Creates new features like date components (Year, Month, DayOfWeek, WeekOfYear), lag sales, and rolling average sales to improve model performance.
*   **Model Training:** Uses a Random Forest Regressor to predict weekly sales.
*   **Model Evaluation:** Evaluates the model using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
*   **Interactive Dashboard (Streamlit):**
    *   Displays predictions on pre-loaded feature-engineered data.
    *   Allows users to filter predictions by Store, Department, Year, and Month.
    *   Option to upload custom feature-engineered CSV for on-the-fly predictions.
    *   Presents Exploratory Data Analysis (EDA) on cleaned data.
    *   Visualizes model feature importances.
*   **Batch Predictions:** Scripts available to generate predictions on a full dataset.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Iamhimanshu008/retail-sales-prediction.git
    cd retail-sales-prediction
    ```

2.  **Install Git LFS (if not already installed):**
    The model file (`model/random_forest_model.pkl`) is tracked using Git LFS due to its size.
    Follow instructions at [https://git-lfs.github.com/](https://git-lfs.github.com/) to install Git LFS for your system.
    After installation, pull LFS files:
    ```bash
    git lfs install
    git lfs pull
    ```
    (If you cloned after LFS was set up, `git lfs pull` might not be needed or `git clone` will handle it).

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure `requirements.txt` is created and up-to-date. See section below.)

## Creating `requirements.txt`

If `requirements.txt` is not present or outdated, you can generate it from your active virtual environment after installing all necessary packages:
```bash
pip freeze > requirements.txt
```
Key libraries used: `pandas`, `scikit-learn`, `streamlit`, `matplotlib`, `seaborn`, `plotly`.

## Running the Project

Follow these steps in order:

1.  **Ensure raw data is in the `data/` directory.**
    (The files `sales data-set.csv`, `stores data-set.csv`, `Features data set.csv` should be present.)

2.  **Run Data Preprocessing:**
    ```bash
    python data_preprocessing.py
    ```
    This generates `output/cleaned_data.csv`.

3.  **Run Feature Engineering:**
    ```bash
    python feature_engineering.py
    ```
    This generates `output/feature_engineered_data.csv`.

4.  **Run Model Training:**
    ```bash
    python model_training.py
    ```
    This generates `model/random_forest_model.pkl` (if not already present and pulled via Git LFS).

5.  **Optional Scripts:**
    *   **Exploratory Data Analysis (Standalone Plots):**
        ```bash
        python eda.py
        ```
        Saves plots to `output/eda_plots/`.
    *   **Feature Importance Plot:**
        ```bash
        python feature_importance.py
        ```
        Saves plot to `output/feature_importance_plot.png`.
    *   **Batch Predictions on Default Data:**
        ```bash
        python model_prediction.py
        ```
        Saves predictions to `output/batch_predicted_sales.csv`.
    *   **Predict on Custom File (Command Line):**
        ```bash
        python predict_uploaded_file.py path/to/your_feature_engineered_file.csv
        ```

6.  **Run the Streamlit Application:**
    ```bash
    streamlit run streamlit_app.py
    ```
    Open your web browser to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Potential Improvements / Future Work

*   Experiment with other regression models (e.g., XGBoost, LightGBM, Neural Networks).
*   Hyperparameter tuning for the chosen model.
*   More advanced feature engineering (e.g., interaction terms, holiday effects).
*   Time series specific models (e.g., ARIMA, Prophet).
*   Deployment to a cloud platform.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

*Developed by Himanshu Shekhar.*
