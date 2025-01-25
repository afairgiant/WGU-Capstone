import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model


def run_ohlc_prediction(data, days):
    """
    Processes OHLC data and predicts future prices using linear regression,
    with enhanced feature engineering, seasonal decomposition, and evaluation.

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
    data["daily_return"] = data["close"].pct_change()

    # Add seasonal decomposition components
    if len(data) >= 730:
        decomposition = seasonal_decompose(data["close"], model="additive", period=365)
        data["trend"] = decomposition.trend
        data["seasonal"] = decomposition.seasonal
        data["residual"] = decomposition.resid
    else:
        print("Not enough data for seasonal decomposition. Skipping this step.")
        data["trend"] = 0
        data["seasonal"] = 0
        data["residual"] = 0

    # Time-Derived Features
    data["day_of_week"] = data["time"].dt.dayofweek
    data["month"] = data["time"].dt.month
    data["year"] = data["time"].dt.year

    # Drop rows with NaN values
    data = data.dropna()
    if data.empty:
        print("Not enough data for prediction after cleaning. Returning empty results.")
        return pd.DataFrame({"Date": [], "Predicted Price": []})

    # Prepare features and target variable
    X = data[
        [
            "timestamp",
            "moving_avg_5",
            "moving_avg_10",
            "daily_return",
            "trend",
            "seasonal",
            "day_of_week",
            "month",
            "year",
        ]
    ]
    y = data["close"]

    # Scale the data
    scaler = MinMaxScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except ValueError as e:
        print(f"Scaling error: {e}")
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
    print(f"Model Mean Squared Error: {mse:.4f}")

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


def lstm_crypto_forecast(data, days):
    """
    Predict future cryptocurrency prices using an LSTM model, with validation, early stopping,
    time-derived features, and enhanced prediction handling.

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

    # Convert time column to datetime and set as index
    data["time"] = pd.to_datetime(data["time"])
    data.set_index("time", inplace=True)

    # Add time-derived features
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["lag_close"] = data["close"].shift(1)  # Lagged close price
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

    if len(X) == 0:  # Handle edge case for small datasets
        raise ValueError("Not enough data to create sequences for LSTM.")

    # Split into training, validation, and test sets
    split_train = int(0.7 * len(X))
    split_val = int(0.85 * len(X))
    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]

    # Build the LSTM model
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(n_steps, X.shape[2])),
            LSTM(64, return_sequences=False),
            Dense(32),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=100,
        verbose=1,
        callbacks=[early_stopping],
    )

    # Save the model
    model.save("lstm_crypto_model.keras")

    # Predict future prices
    future_predictions = []
    last_sequence = scaled_data[-n_steps:, :]  # Last sequence of features
    for _ in range(days):
        last_sequence_reshaped = last_sequence.reshape(
            (1, n_steps, last_sequence.shape[1])
        )
        next_prediction = model.predict(last_sequence_reshaped, verbose=0)
        future_predictions.append(next_prediction[0, 0])

        # Update the last_sequence
        next_input = last_sequence[1:]  # Remove the first row (shift the sequence)
        new_row = last_sequence[-1].copy()  # Copy the last row as a template
        new_row[0] = next_prediction[
            0, 0
        ]  # Update the `close` feature with the prediction
        last_sequence = np.vstack([next_input, new_row])  # Append the new row

    # Transform predictions back to the original scale
    future_predictions_expanded = np.zeros(
        (len(future_predictions), scaled_data.shape[1])
    )
    future_predictions_expanded[:, 0] = (
        future_predictions  # Assign predictions to the `close` feature
    )
    future_predictions = scaler.inverse_transform(future_predictions_expanded)[
        :, 0
    ]  # Extract the `close` column

    # Generate future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(last_date, periods=days + 1, freq="D")[1:]

    # Create a DataFrame for predictions
    predictions = pd.DataFrame(
        {"Date": future_dates, "Predicted Price": future_predictions.flatten()}
    )

    return predictions


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


def plot_correlation_heatmap(data, target_column=None, highlight_threshold=0.8):
    """
    Generate and display a correlation heatmap with scaling, annotation, and optional highlighting.

    Args:
        data (pd.DataFrame): DataFrame containing numerical columns.
        target_column (str, optional): Highlight correlations with this target column.
        highlight_threshold (float): Threshold for highlighting strong correlations.

    Returns:
        matplotlib.figure.Figure: The heatmap figure.
    """
    # Select numeric data only
    numeric_data = data.select_dtypes(include=["number"])

    # Ensure there are enough numeric columns
    if numeric_data.empty or numeric_data.shape[1] < 2:
        raise ValueError("Not enough numerical data to generate a heatmap.")

    # Scale the numeric data for consistency
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(numeric_data), columns=numeric_data.columns
    )

    # Calculate correlation matrix
    correlation_matrix = scaled_data.corr()

    # Highlight strong correlations
    if target_column and target_column in correlation_matrix.columns:
        target_corr = correlation_matrix[target_column].sort_values(ascending=False)
        strong_corrs = target_corr[
            (target_corr >= highlight_threshold) | (target_corr <= -highlight_threshold)
        ]
        print(f"Strong correlations with {target_column}:\n{strong_corrs}")

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"label": "Correlation Coefficient"},
        ax=ax,
    )

    # Add title and labels
    ax.set_title("Feature Correlation Heatmap", fontsize=16)
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)

    # Optionally annotate specific correlations
    if target_column:
        for i, column in enumerate(correlation_matrix.columns):
            if (
                correlation_matrix.loc[target_column, column] >= highlight_threshold
                or correlation_matrix.loc[target_column, column] <= -highlight_threshold
            ):
                ax.text(
                    i + 0.5,  # X-coordinate (column index)
                    correlation_matrix.columns.get_loc(target_column) + 0.5,  # Y-index
                    f"{correlation_matrix.loc[target_column, column]:.2f}",
                    color="black",
                    ha="center",
                    va="center",
                    fontsize=10,
                    bbox=dict(facecolor="yellow", alpha=0.5),
                )

    return fig
