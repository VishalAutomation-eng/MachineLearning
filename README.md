Project Title:

Time Series Forecasting for Stock Price Prediction

Description:

This project implements time series forecasting techniques to predict stock prices and trends. It leverages historical stock data to build models that can forecast future values. The primary focus is on understanding time series patterns like trend, seasonality, and residuals, and using machine learning to generate accurate forecasts.

The project uses Tesla (TSLA) stock price data from Yahoo Finance as a real-world dataset.

Key Features:

Fetching real-time stock data using the yfinance API.

Data cleaning and handling missing timestamps.

Time series decomposition into trend, seasonality, and noise.

Feature engineering:

Lag features

Rolling mean/standard deviation

Calendar-based features (day, month, weekday)

Splitting dataset chronologically (no shuffling).

Training models:

Linear Regression

Random Forest Regressor

(Optional) XGBoost for boosting performance

Model evaluation using time series metrics:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

MAPE (Mean Absolute Percentage Error)

R² (Coefficient of Determination)

Visualization of actual vs predicted values for better insights.
