1. ASHRAE Energy Prediction Project
Project Title:

ASHRAE - Energy Usage Prediction using Machine Learning

Description:

The ASHRAE Energy Prediction Project focuses on predicting building energy consumption based on historical data using machine learning algorithms. The project leverages datasets from the ASHRAE - Great Energy Predictor III competition hosted on Kaggle, which contains information about energy meter readings, building characteristics, weather data, and more.

The primary goal is to develop accurate predictive models to forecast energy usage (kWh, m³, etc.) for multiple buildings, which can help reduce energy waste, optimize building operations, and improve sustainability.

By applying time series forecasting and regression techniques, this project provides valuable insights into energy consumption patterns, enabling facility managers and engineers to make data-driven decisions for energy efficiency improvements.




2. Single Zone Variable Air Volume (SZVAV) System
Overview:

The Single Zone Variable Air Volume (SZVAV) system is a type of HVAC (Heating, Ventilation, and Air Conditioning) control system designed to efficiently manage air distribution and temperature in a single thermal zone.

Unlike traditional Constant Volume (CV) systems, where the airflow remains constant and only the supply air temperature changes, SZVAV systems adjust the airflow rate in response to the thermal load of the space. This allows for better energy efficiency, comfort control, and cost savings, especially in buildings with variable occupancy or load patterns.

Key Features:

Single Thermal Zone:

Ideal for spaces with uniform heating and cooling requirements.

Examples:

Small office floors

Conference rooms

Retail spaces

Classrooms

Variable Air Volume Control:

Automatically adjusts fan speed and airflow based on:

Room temperature

Cooling/heating demand

Occupancy levels

Energy Savings:

By reducing fan speed during low load conditions, energy consumption decreases.

Aligns with ASHRAE 90.1 energy efficiency standards.

Integration with Building Automation Systems (BAS):

Can be integrated with IoT sensors and smart controls for real-time optimization.

Components of SZVAV System:

Supply Fan:
Variable speed fan that adjusts airflow according to demand.

Dampers:
Control air distribution to maintain desired temperature.

Temperature Sensors:
Detect room temperature to adjust fan speed and cooling/heating.

Controller/Actuator:
Uses control logic to modulate airflow and temperature.

Heating and Cooling Coil:
Conditions the air before it enters the occupied space.

How SZVAV Works:

Load Detection:

Sensors measure the room temperature and compare it with the setpoint.

Fan Adjustment:

If the room is near the set temperature:

The fan speed decreases → reducing airflow.

If the room is far from the set temperature:

The fan speed increases → providing more conditioned air.

Energy Optimization:

During partial load conditions, the system minimizes energy consumption by reducing both fan power and cooling/heating energy.






3. Project Title:

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
