import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


def run_ohlc_prediction(data, days):
    """
    Processes OHLC data and predicts future prices using linear regression.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC data.
        days (int): Number of future days to predict.

    Returns:
        pd.DataFrame: A DataFrame containing dates and predicted prices.
    """
    # Ensure required columns exist
    required_columns = ["time", "open", "high", "low", "close"]
    if not all(col in data.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in data.columns]
        raise ValueError(f"The data is missing columns: {missing_cols}")

    # Convert time column to datetime
    data["time"] = pd.to_datetime(data["time"])

    # Sort data by time
    data = data.sort_values(by="time")

    # Feature Engineering
    data["timestamp"] = data["time"].apply(lambda x: x.timestamp())
    data["moving_avg_5"] = (
        data["close"].rolling(window=5).mean()
    )  # 5-period moving average
    data["moving_avg_10"] = (
        data["close"].rolling(window=10).mean()
    )  # 10-period moving average
    data["daily_return"] = data["close"].pct_change()  # Daily percentage return
    data = data.dropna()  # Drop rows with NaN values from rolling calculations

    # Prepare features and target variable
    X = data[["timestamp", "moving_avg_5", "moving_avg_10", "daily_return"]]
    y = data["close"]

    # Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Predict future prices
    last_time = data["time"].iloc[-1]
    future_dates = [last_time + timedelta(days=i) for i in range(1, days + 1)]

    # Generate future feature data
    future_data = pd.DataFrame(
        {
            "timestamp": [d.timestamp() for d in future_dates],
            "moving_avg_5": [data["moving_avg_5"].iloc[-1]] * days,
            "moving_avg_10": [data["moving_avg_10"].iloc[-1]] * days,
            "daily_return": [data["daily_return"].iloc[-1]] * days,
        }
    )
    future_scaled = scaler.transform(future_data)

    # Predict
    future_predictions = model.predict(future_scaled)

    # Combine dates and predictions into a DataFrame
    predictions = pd.DataFrame(
        {"Date": future_dates, "Predicted Price": future_predictions}
    )
    return predictions


def calculate_daily_average(data):
    """
    Calculates the daily average price for the given OHLC data.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC data.

    Returns:
        pd.DataFrame: A DataFrame containing dates and their corresponding daily average prices.
    """
    # Ensure required columns exist
    required_columns = ["time", "open", "high", "low", "close"]
    if not all(col in data.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in data.columns]
        raise ValueError(f"The data is missing columns: {missing_cols}")

    # Convert time column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(data["time"]):
        data["time"] = pd.to_datetime(data["time"])

    # Debugging: Check the data
    print("Data after ensuring datetime format:", data.head())

    # Extract the date part from the datetime
    data["date"] = data["time"].dt.date

    # Debugging: Check if 'date' column is added
    print("Data with 'date' column:", data.head())

    # Calculate the daily average price
    data["daily_average"] = data[["open", "high", "low", "close"]].mean(axis=1)

    # Group by date and calculate the mean of daily averages for each date
    daily_avg = data.groupby("date")["daily_average"].mean().reset_index()

    # Rename columns for clarity
    daily_avg.columns = ["Date", "Average Price"]

    # Debugging: Check the output
    print("Daily averages calculated:", daily_avg.head())

    return daily_avg
