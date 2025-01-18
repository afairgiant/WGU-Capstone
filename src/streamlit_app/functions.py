import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


def run_ohlc_prediction(data, days):
    """
    Processes OHLC data and predicts future prices using linear regression,
    with enhanced feature engineering including time-derived features.

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
    data["moving_avg_5"] = data["close"].rolling(window=5, min_periods=1).mean()
    data["moving_avg_10"] = data["close"].rolling(window=10, min_periods=1).mean()
    data["daily_return"] = data["close"].pct_change()  # Daily percentage return

    # Time-Derived Features
    data["day_of_week"] = data["time"].dt.dayofweek
    data["month"] = data["time"].dt.month
    data["year"] = data["time"].dt.year

    # Drop rows with NaN values from rolling calculations
    data = data.dropna()

    # Prepare features and target variable
    X = data[
        [
            "timestamp",
            "moving_avg_5",
            "moving_avg_10",
            "daily_return",
            "day_of_week",
            "month",
            "year",
        ]
    ]
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
            "day_of_week": [d.dayofweek for d in future_dates],
            "month": [d.month for d in future_dates],
            "year": [d.year for d in future_dates],
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


def lstm_crypto_forecast(data, days):
    """
    Predict future cryptocurrency prices using an LSTM model.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC data.
        days (int): Number of future days to predict.

    Returns:
        pd.DataFrame: A DataFrame containing dates and predicted prices.
    """

    # Ensure required columns exist
    required_columns = ["time", "close"]
    if not all(col in data.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in data.columns]
        raise ValueError(f"The data is missing columns: {missing_cols}")

    # Convert time column to datetime
    data["time"] = pd.to_datetime(data["time"])
    data.set_index("time", inplace=True)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[["close"]])

    # Create sequences for LSTM
    def create_sequences(data, n_steps):
        X, y = [], []
        for i in range(n_steps, len(data)):
            X.append(data[i - n_steps : i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    n_steps = 30  # Use the last 30 days to predict the next day
    X, y = create_sequences(scaled_data, n_steps)

    # Reshape for LSTM input
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split into training and test sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build the LSTM model
    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=(n_steps, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1)

    # Predict future prices
    future_predictions = []
    last_sequence = scaled_data[-n_steps:]
    for _ in range(days):
        # Reshape the last sequence for prediction
        last_sequence_reshaped = last_sequence.reshape((1, n_steps, 1))
        next_prediction = model.predict(last_sequence_reshaped, verbose=0)
        future_predictions.append(next_prediction[0, 0])

        # Update the last sequence
        last_sequence = np.append(last_sequence[1:], [[next_prediction[0, 0]]], axis=0)

    # Transform predictions back to the original scale
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    # Generate future dates
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]

    # Create a DataFrame for predictions
    predictions = pd.DataFrame(
        {"Date": future_dates, "Predicted Price": future_predictions.flatten()}
    )
    return predictions
