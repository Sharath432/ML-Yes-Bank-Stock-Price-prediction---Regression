Yes Bank Stock Closing Price Prediction
Overview
This project aims to predict the monthly closing stock prices of Yes Bank, a major Indian financial institution, using historical stock price data. The project employs machine learning techniques to build a robust regression model, with the goal of supporting trading and investment decisions. The dataset includes monthly Open, High, Low, and Close (OHLC) prices, and the final model achieves high accuracy with an RMSE of 13.14 and R² of 0.98.
Key Features:

Comprehensive data exploration and visualization.
Feature engineering to capture price dynamics and trends.
Implementation of Linear Regression, Random Forest, and XGBoost models.
Model interpretability using SHAP for feature importance.
Business-focused insights for trading and risk management.

Table of Contents

Dataset
Installation
Usage
Project Structure
Methodology
Results
Future Work
Contributing
License
Contact

Dataset
The dataset (data_YesBank_StockPrices.csv) contains monthly stock price data for Yes Bank, with the following columns:

Date: Month of the data (e.g., MMM-YY).
Open: Opening price of the month.
High: Highest price of the month.
Low: Lowest price of the month.
Close: Closing price of the month (target variable).

Key Characteristics:

No missing values or duplicates.
Outliers identified, reflecting market events (e.g., fraud case).
Date set as index for time series analysis.


Note: The dataset is not included in this repository due to size constraints. You can download it from [source link, if applicable] or contact the repository owner.

Installation
To run this project locally, follow these steps:

Clone the Repository:
git clone https://github.com/your-username/yes-bank-stock-prediction.git
cd yes-bank-stock-prediction


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Install the required Python libraries listed in requirements.txt:
pip install -r requirements.txt

Key Dependencies:

pandas, numpy: Data manipulation.
matplotlib, seaborn, mplfinance: Visualization.
scikit-learn: Machine learning models.
xgboost: XGBoost model.
shap: Model interpretability.
statsmodels: Time series analysis (future work).


Add the Dataset:Place data_YesBank_StockPrices.csv in the data/ directory or update the file path in the notebook.


Usage

Run the Jupyter Notebook:Launch Jupyter Notebook and open the main project file:
jupyter notebook Yes_Bank_Stock_Closing_Price_Prediction.ipynb


Follow the Notebook:

The notebook is structured into sections: data exploration, visualization, feature engineering, modeling, and evaluation.
Execute cells sequentially to reproduce the analysis and results.
Key outputs include visualizations, model performance metrics (RMSE, R²), and SHAP feature importance plots.


Predict on New Data:

The trained XGBoost model can be used for predictions on new data (requires preprocessing as per the notebook).
Model saving/loading is planned for future implementation (see Future Work).



Project Structure
yes-bank-stock-prediction/
├── data/
│   └── data_YesBank_StockPrices.csv  # Dataset (not included)
├── Yes_Bank_Stock_Closing_Price_Prediction.ipynb  # Main notebook
├── requirements.txt  # Python dependencies
├── README.md  # This file
└── outputs/  # Directory for saving plots and models (optional)

Methodology
The project follows a structured machine learning pipeline:

Data Exploration:

Loaded and inspected the dataset for missing values, duplicates, and outliers.
Set Date as the index for time series compatibility.


Data Visualization:

Created 11 charts (e.g., box plots, candlestick, moving averages) following the UBM (Univariate, Bivariate, Multivariate) framework.
Insights: Identified trends, volatility, and outliers (e.g., due to market events).


Feature Engineering:

Created features: Price_Change_Pct, Volatility_Ratio, Close_Lag1, Close_Rolling_Std_3, MA10, MA20, Is_Green_Month.
Selected features using Variance Inflation Factor (VIF) and statistical tests to reduce multicollinearity.


Data Transformation:

Applied Log and Power transformations to handle skewness.
Used StandardScaler for feature scaling.
Differenced Close for stationarity (for potential time series models).


Data Splitting:

Split data into 80% training and 20% testing (random_state=42).


Modeling:

Implemented three models:
Linear Regression (Ridge): Tuned alpha with GridSearchCV.
Random Forest Regressor: Tuned n_estimators, max_depth, etc.
XGBoost Regressor: Tuned n_estimators, learning_rate, max_depth, subsample.


Evaluated using RMSE and R².


Model Interpretability:

Used SHAP to analyze feature importance (e.g., Low, High, Avg_Price were key drivers).



Results

Best Model: Tuned XGBoost Regressor.
Performance:
RMSE: 13.14 (average prediction error of ~₹13).
R²: 0.98 (explains 98% of variance in Close prices).
Mean CV R²: ~0.97 (indicates robust generalization).


Feature Importance (via SHAP on Random Forest):
Most influential: Low, Avg_Price, High.
Less impactful: Price_Change_Pct, MA10, Volatility_Ratio.


Business Impact:
Low RMSE ensures precise predictions, minimizing trading risks.
High R² builds confidence in model reliability for investment strategies.
Visualizations and SHAP insights guide trading decisions by highlighting trends and key price drivers.



Future Work

Complete Visualizations: Add 4 more charts (e.g., correlation heatmap, volatility trends) to meet the 15-chart goal.
Implement Time Series Models: Test ARIMA, SARIMA, or Prophet for comparison with machine learning models.
Handle Outliers: Apply winsorization or robust scaling to reduce outlier impact.
Model Deployment: Save the XGBoost model using pickle or joblib and test on unseen data.
Extend SHAP Analysis: Apply SHAP to the XGBoost model for consistent feature importance insights.
Additional Metrics: Include Mean Absolute Error (MAE) and directional accuracy for comprehensive evaluation.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make changes and commit (git commit -m "Add feature").
Push to your fork (git push origin feature-branch).
Open a pull request with a detailed description of changes.

Please ensure code follows PEP 8 guidelines and includes relevant tests.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, reach out to:

Author: Sarathraj R
Email: rsarath16new@gmail.com

