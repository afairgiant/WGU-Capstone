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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model


def run_ohlc_prediction(data, days):
    """
    Predict future prices using a linear regression model.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC data.
        days (int): Number of future days to predict.

    Returns:
        pd.DataFrame: A DataFrame containing predicted future prices.
    """

    # Ensure the 'time' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data["time"]):
        data["time"] = pd.to_datetime(data["time"], errors="coerce")

    # Drop rows with invalid or NaT values after conversion
    data = data.dropna(subset=["time"])

    # Feature engineering
    data["timestamp"] = data["time"].apply(lambda x: x.timestamp())
    data["moving_avg_5"] = data["close"].rolling(window=5, min_periods=1).mean()
    data["moving_avg_10"] = data["close"].rolling(window=10, min_periods=1).mean()
    data["daily_return"] = data["close"].pct_change()
    data.dropna(inplace=True)

    # Features and target
    X = data[["timestamp", "moving_avg_5", "moving_avg_10", "daily_return"]]
    y = data["close"]

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future prices
    last_row = data.iloc[-1]
    future_timestamps = [
        last_row["timestamp"] + i * 86400 for i in range(1, days + 1)
    ]  # 86400 seconds in a day
    future_data = pd.DataFrame(
        {
            "timestamp": future_timestamps,
            "moving_avg_5": [last_row["moving_avg_5"]] * days,
            "moving_avg_10": [last_row["moving_avg_10"]] * days,
            "daily_return": [last_row["daily_return"]] * days,
        }
    )
    future_scaled = scaler.transform(future_data)
    future_predictions = model.predict(future_scaled)

    # Return predictions as a DataFrame
    predictions = pd.DataFrame(
        {
            "Date": pd.to_datetime(future_timestamps, unit="s"),
            "Predicted Price": future_predictions,
        }
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


def lstm_crypto_forecast(model, scaler, data, days):
    try:
        # Rename 'date' to 'time' if needed
        if "time" not in data.columns and "date" in data.columns:
            data.rename(columns={"date": "time"}, inplace=True)
            st.write("Renamed 'date' column to 'time'.")

        # Validate 'time' column
        if "time" not in data.columns:
            raise ValueError("Input data must contain a 'time' column.")

        # Convert 'time' to datetime and drop invalid rows
        data["time"] = pd.to_datetime(data["time"], errors="coerce")
        st.write("Converted 'time' column to datetime format.")
        data.dropna(subset=["time"], inplace=True)

        # Set 'time' as the index
        data.set_index("time", inplace=True)
        st.write("Data after setting 'time' as index:", data.head())

        # Feature engineering
        data["day_of_week"] = data.index.dayofweek
        data["month"] = data.index.month
        data["lag_close"] = data["close"].shift(1)
        data.dropna(inplace=True)
        st.write("Data after feature engineering:", data.head())

        # Scale features
        features = ["close", "day_of_week", "month", "lag_close"]
        scaled_data = scaler.transform(data[features])
        st.write("Scaled data shape:", scaled_data.shape)

        # Check the last sequence
        n_steps = 30
        last_sequence = scaled_data[-n_steps:, :]
        st.write("Last sequence for prediction:", last_sequence)

        # Generate future predictions
        future_predictions = []
        for i in range(days):
            last_sequence_reshaped = last_sequence.reshape(
                (1, n_steps, last_sequence.shape[1])
            )
            next_prediction = model.predict(last_sequence_reshaped, verbose=0)[0, 0]
            future_predictions.append(next_prediction)

            # Update the sequence
            last_sequence = np.vstack(
                [last_sequence[1:], [next_prediction] + last_sequence[-1, 1:].tolist()]
            )

        # Scale back predictions
        future_predictions_expanded = np.zeros(
            (len(future_predictions), scaled_data.shape[1])
        )
        future_predictions_expanded[:, 0] = future_predictions
        future_predictions = scaler.inverse_transform(future_predictions_expanded)[:, 0]

        # Generate future dates using the index
        future_dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1), periods=days
        )
        st.write("Future dates for predictions:", future_dates)

        # Create a DataFrame for predictions
        predictions = pd.DataFrame(
            {"Date": future_dates, "Predicted Price": future_predictions}
        )

        return predictions

    except Exception as e:
        st.error(f"Error during predictions: {e}")
        raise


def train_lstm_model(data):
    """
    Train an LSTM model for cryptocurrency price prediction.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC data.

    Returns:
        tuple: A tuple containing:
            - model: Trained LSTM model.
            - scaler: Fitted scaler for feature scaling.
            - X_test (np.ndarray): Test features for evaluation.
            - y_test (np.ndarray): Test target values for evaluation.
    """

    # Ensure the 'time' column is in datetime format
    if "time" not in data.columns:
        raise ValueError("Input data must contain a 'time' column.")
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data.dropna(subset=["time"], inplace=True)
    data.set_index("time", inplace=True)

    # Add time-derived features
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["lag_close"] = data["close"].shift(1)
    data.dropna(inplace=True)

    # Prepare features and target
    features = ["close", "day_of_week", "month", "lag_close"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    n_steps = 30

    def create_sequences(data, n_steps):
        X, y = [], []
        for i in range(n_steps, len(data)):
            X.append(data[i - n_steps : i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, n_steps)

    # Train-test split
    split_train = int(0.7 * len(X))
    split_val = int(0.85 * len(X))
    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]

    # Define LSTM model
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(n_steps, X.shape[2])),
            LSTM(64),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Train model
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=100,
        callbacks=[early_stopping],
        verbose=1,
    )

    return model, scaler, X_test, y_test


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


def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of a model using various metrics.

    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        dict: A dictionary containing MAE, RMSE, and R² metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)  # MSE
    rmse = np.sqrt(mse)  # RMSE manually calculated
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R²": r2}


def evaluate_lstm_model(model, X_test, y_test):
    """
    Evaluate the performance of an LSTM model.

    Args:
        model: Trained LSTM model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True target values.

    Returns:
        dict: A dictionary containing evaluation metrics (MAE, RMSE, R²).
    """
    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = evaluate_model(
        y_test, y_pred
    )  # Uses your existing evaluate_model function
    return metrics


def evaluate_linear_regression_model(data):
    """
    Evaluate the performance of a linear regression model on historical data.

    Args:
        data (pd.DataFrame): DataFrame containing historical OHLC data.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # Ensure the 'time' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data["time"]):
        data["time"] = pd.to_datetime(data["time"], errors="coerce")

    # Drop rows with invalid or NaT values after conversion
    data = data.dropna(subset=["time"])

    # Feature engineering
    data["timestamp"] = data["time"].apply(lambda x: x.timestamp())
    data["moving_avg_5"] = data["close"].rolling(window=5, min_periods=1).mean()
    data["moving_avg_10"] = data["close"].rolling(window=10, min_periods=1).mean()
    data["daily_return"] = data["close"].pct_change()
    data.dropna(inplace=True)

    # Features and target
    X = data[["timestamp", "moving_avg_5", "moving_avg_10", "daily_return"]]
    y = data["close"]

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    metrics = evaluate_model(
        y_test, y_pred
    )  # Uses your existing `evaluate_model` function
    return metrics
