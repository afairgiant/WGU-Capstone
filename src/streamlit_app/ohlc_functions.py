import logging
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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses OHLC data by converting time, sorting, and adding basic features.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.

    Raises:
        ValueError: If required columns are missing or data is empty.
    """
    # Ensure required columns exist
    required_columns = ["time", "open", "high", "low", "close"]
    if not all(col in data.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in data.columns]
        raise ValueError(f"The data is missing columns: {missing_cols}")

    if data.empty:
        raise ValueError("The input data is empty.")

    # Convert time column to datetime
    data["time"] = pd.to_datetime(data["time"], errors="coerce")

    # Check for duplicate timestamps
    if data["time"].duplicated().any():
        raise ValueError("Duplicate timestamps found in the data.")

    # Sort data by time
    data = data.sort_values(by="time").reset_index(drop=True)

    # Feature Engineering
    data["moving_avg_5"] = data["close"].rolling(window=5, min_periods=1).mean()
    data["moving_avg_10"] = data["close"].rolling(window=10, min_periods=1).mean()
    data["daily_return"] = data["close"].pct_change()

    # Time-Derived Features
    data["day_of_week"] = data["time"].dt.dayofweek
    data["month"] = data["time"].dt.month
    data["year"] = data["time"].dt.year

    return data


def add_seasonal_decomposition(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds seasonal decomposition components to the data.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC data.

    Returns:
        pd.DataFrame: DataFrame with added seasonal decomposition components.
    """
    if len(data) < 730:
        print("Not enough data for seasonal decomposition. Skipping this step.")
        data["trend"] = 0
        data["seasonal"] = 0
        data["residual"] = 0
        return data

    try:
        decomposition = seasonal_decompose(data["close"], model="additive", period=365)
        data["trend"] = decomposition.trend
        data["seasonal"] = decomposition.seasonal
        data["residual"] = decomposition.resid
    except Exception as e:
        print(f"Error during seasonal decomposition: {e}")
        data["trend"] = 0
        data["seasonal"] = 0
        data["residual"] = 0

    return data


def run_ohlc_prediction(data: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Processes OHLC data and predicts future prices using linear regression.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC data.
        days (int): Number of future days to predict.

    Returns:
        pd.DataFrame: A DataFrame containing dates and predicted prices.
    """
    # Preprocess data
    data = preprocess_data(data)
    data = add_seasonal_decomposition(data)

    # Drop rows with NaN values
    data = data.dropna()
    if data.empty:
        logging.warning(
            "Not enough data for prediction after cleaning. Returning empty results."
        )
        return pd.DataFrame({"Date": [], "Predicted Price": []})

    # Prepare features and target variable
    features = [
        "moving_avg_5",
        "moving_avg_10",
        "daily_return",
        "trend",
        "seasonal",
        "day_of_week",
        "month",
        "year",
    ]
    X = data[features]
    y = data["close"]

    # Scale the data
    scaler = MinMaxScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except ValueError as e:
        logging.error(f"Scaling error: {e}")
        return pd.DataFrame({"Date": [], "Predicted Price": []})

    # Train-Test Split for Model Evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Model Mean Squared Error: {mse:.4f}")

    # Predict future prices
    last_time = data["time"].iloc[-1]
    future_dates = [last_time + timedelta(days=i) for i in range(1, days + 1)]

    # Generate future feature data
    future_data = pd.DataFrame(
        {
            "moving_avg_5": [data["moving_avg_5"].iloc[-1]] * days,
            "moving_avg_10": [data["moving_avg_10"].iloc[-1]] * days,
            "daily_return": [data["daily_return"].iloc[-1]] * days,
            "trend": [data["trend"].iloc[-1]] * days,
            "seasonal": [data["seasonal"].iloc[-1]] * days,
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


def lstm_crypto_forecast(data: pd.DataFrame, days: int) -> pd.DataFrame:
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

    if len(data) < 30:
        raise ValueError(
            "Insufficient data for LSTM training. At least 30 rows are required."
        )

    # Preprocess data
    data["time"] = pd.to_datetime(data["time"])
    data.set_index("time", inplace=True)

    # Add time-derived features
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["lag_close"] = data["close"].shift(1)
    data = data.dropna()

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(
        data[["close", "day_of_week", "month", "lag_close"]]
    )

    # Create sequences for LSTM
    def create_sequences(data, n_steps):
        X, y = [], []
        for i in range(n_steps, len(data)):
            X.append(data[i - n_steps : i])
            y.append(data[i, 0])  # Predict "close" price
        return np.array(X), np.array(y)

    n_steps = 30
    X, y = create_sequences(scaled_data, n_steps)

    # Split into training, validation, and test sets
    split_train = int(0.7 * len(X))
    split_val = int(0.85 * len(X))
    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]

    # Train the model
    try:
        model = train_lstm_model(X_train, y_train, X_val, y_val, n_steps, X.shape[2])
    except Exception as e:
        logging.error(f"Error during LSTM model training: {e}")
        return pd.DataFrame({"Date": [], "Predicted Price": []})

    # Predict future prices
    future_predictions = predict_future_prices(
        model, scaled_data, scaler, n_steps, days
    )

    # Generate future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(last_date, periods=days + 1, freq="D")[1:]

    # Create a DataFrame for predictions
    predictions = pd.DataFrame(
        {"Date": future_dates, "Predicted Price": future_predictions.flatten()}
    )

    return predictions


def train_lstm_model(X_train, y_train, X_val, y_val, n_steps, n_features):
    """
    Train an LSTM model with the given data.

    Args:
        X_train (np.ndarray): Training feature sequences.
        y_train (np.ndarray): Training target values.
        X_val (np.ndarray): Validation feature sequences.
        y_val (np.ndarray): Validation target values.
        n_steps (int): Number of time steps in the input sequence.
        n_features (int): Number of features per time step.

    Returns:
        keras.Model: Trained LSTM model.
    """
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(n_steps, n_features)),
            LSTM(64, return_sequences=False),
            Dense(32),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=100,
        verbose=1,
        callbacks=[early_stopping],
    )

    model.save("lstm_crypto_model.keras")
    return model


def predict_future_prices(model, scaled_data, scaler, n_steps, days):
    """
    Predict future prices using a trained LSTM model.

    Args:
        model (keras.Model): Trained LSTM model.
        scaled_data (np.ndarray): Scaled data array.
        scaler (MinMaxScaler): Fitted scaler for inverse transformation.
        n_steps (int): Number of time steps in the input sequence.
        days (int): Number of future days to predict.

    Returns:
        np.ndarray: Array of predicted prices.
    """
    future_predictions = []
    last_sequence = scaled_data[-n_steps:, :]
    for _ in range(days):
        last_sequence_reshaped = last_sequence.reshape(
            (1, n_steps, last_sequence.shape[1])
        )
        next_prediction = model.predict(last_sequence_reshaped, verbose=0)
        future_predictions.append(next_prediction[0, 0])

        # Update the last_sequence
        next_input = last_sequence[1:]
        new_row = last_sequence[-1].copy()
        new_row[0] = next_prediction[0, 0]
        last_sequence = np.vstack([next_input, new_row])

    future_predictions_expanded = np.zeros(
        (len(future_predictions), scaled_data.shape[1])
    )
    future_predictions_expanded[:, 0] = future_predictions
    future_predictions = scaler.inverse_transform(future_predictions_expanded)[:, 0]

    return future_predictions


def calculate_moving_averages(file):
    """
    Processes a CSV file with columns 'time', 'open', 'high', 'low', 'close'.
    Returns a DataFrame with moving averages calculated for the data using daily average price.

    Parameters:
        file: str or file-like object
            Path to the CSV file or file object.

    Returns:
        pd.DataFrame: A DataFrame containing time, daily average price, 7-day moving average, and 30-day moving average.
    """
    # Read the CSV file
    df = pd.read_csv(file)

    # Ensure required columns are present
    required_columns = {"time", "open", "high", "low", "close"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"The CSV file must contain the following columns: {required_columns}"
        )

    # Convert time column to datetime
    df["time"] = pd.to_datetime(df["time"])

    # Sort by time
    df = df.sort_values(by="time").reset_index(drop=True)

    # Calculate the daily average price
    df["average_price"] = df[["open", "high", "low", "close"]].mean(axis=1)

    # Calculate moving averages based on the daily average price
    df["7_day_MA"] = df["average_price"].rolling(window=7).mean()
    df["30_day_MA"] = df["average_price"].rolling(window=30).mean()

    # Prepare the result DataFrame
    moving_averages = df[["time", "average_price", "7_day_MA", "30_day_MA"]]
    moving_averages.columns = [
        "Time",
        "Daily Average Price",
        "7-Day Moving Average",
        "30-Day Moving Average",
    ]

    return moving_averages


def analyze_prices_by_day(file):
    """
    Analyzes whether the price typically goes up or down on each day of the week.

    Parameters:
        file: str or file-like object
            Path to the CSV file or file object.

    Returns:
        pd.Series: A Series containing the average daily price change for each day of the week.
    """
    # Read the CSV file
    df = pd.read_csv(file)

    # Ensure required columns are present
    required_columns = {"time", "open", "high", "low", "close"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"The CSV file must contain the following columns: {required_columns}"
        )

    # Convert time column to datetime
    df["time"] = pd.to_datetime(df["time"])

    # Sort by time to ensure proper calculation of changes
    df = df.sort_values(by="time").reset_index(drop=True)

    # Calculate daily price change (close - open)
    df["daily_change"] = df["close"] - df["open"]

    # Extract day of the week
    df["day_of_week"] = df["time"].dt.day_name()

    # Group by day of the week and calculate the average daily change
    day_avg_change = df.groupby("day_of_week")["daily_change"].mean()

    # Ensure the order of days is correct
    day_avg_change = day_avg_change.reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    return day_avg_change
