import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from utils import setup_logging

# Set up logging
setup_logging()


def preprocess_market_data(df):
    """
    Preprocess the market data by adding new features and scaling.

    Args:
        df (pd.DataFrame): DataFrame containing 'price', 'market_cap', and 'total_volume'.

    Returns:
        pd.DataFrame, MinMaxScaler: Processed DataFrame with additional features and the scaler object.
    """
    logging.info("Preprocessing market data...")
    # Ensure time is a datetime object
    df["time"] = pd.to_datetime(df["time"])

    # Add percentage change features
    df["price_return"] = df["price"].pct_change()
    df["volume_change"] = df["total_volume"].pct_change()
    df["market_cap_change"] = df["market_cap"].pct_change()

    # Add rolling statistics
    df["price_7_day_ma"] = df["price"].rolling(window=7).mean()
    df["price_30_day_ma"] = df["price"].rolling(window=30).mean()
    df["volatility_7_day"] = df["price_return"].rolling(window=7).std()

    # Add ratio features
    df["market_cap_to_volume"] = df["market_cap"] / df["total_volume"]

    # Drop NaN rows introduced by rolling calculations
    df = df.dropna()

    # Scale the features
    features = [
        "price",
        "total_volume",
        "market_cap",
        "price_return",
        "volume_change",
        "market_cap_change",
        "volatility_7_day",
    ]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    logging.info("Market data preprocessing complete.")
    return df, scaler


def create_sequences(data, seq_length):
    """
    Create sequences for time series data to be used in LSTM models.

    Args:
        data (np.array): Array of feature data.
        seq_length (int): Number of time steps in each sequence.

    Returns:
        np.array, np.array: Sequences (X) and target values (y).
    """
    logging.info(f"Creating sequences with sequence length {seq_length}...")
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
        targets.append(data[i + seq_length, 0])  # Assuming 'price' is the target
    return np.array(sequences), np.array(targets)


def prepare_lstm_data(df, seq_length):
    """
    Prepare data for LSTM model training and testing.

    Args:
        df (pd.DataFrame): DataFrame containing market data.
        seq_length (int): Number of time steps in each sequence.

    Returns:
        np.array, np.array, np.array, np.array: X_train, X_test, y_train, y_test.
    """
    # Split into features and target
    features = df[
        [
            "price",
            "total_volume",
            "market_cap",
            "price_return",
            "volume_change",
            "market_cap_change",
            "volatility_7_day",
        ]
    ].values

    # Create sequences
    X, y = create_sequences(features, seq_length)

    # Split into train and test sets (80/20 split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


def build_lstm_model(input_shape):
    """
    Build an LSTM model.

    Args:
        input_shape (tuple): Shape of input data.

    Returns:
        keras.Model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate the LSTM model's performance on test data.

    Args:
        model: Trained LSTM model.
        X_test (np.array): Test data sequences.
        y_test (np.array): True target values.
        scaler (MinMaxScaler): Scaler used for data normalization.

    Returns:
        None
    """
    # Predict on test data
    predictions = model.predict(X_test)

    # Extract the scaler's min and scale for the target feature (price)
    price_min = scaler.min_[0]
    price_scale = scaler.scale_[0]

    # Rescale predictions and true values to their original scale
    predictions_rescaled = predictions * price_scale + price_min
    y_test_rescaled = y_test.reshape(-1, 1) * price_scale + price_min

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Plot predictions vs true values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label="True Values")
    plt.plot(predictions_rescaled, label="Predictions")
    plt.title("LSTM Model Predictions vs True Values")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def plot_market_trends(df):
    """
    Plot trends for price, market capitalization, and volume.

    Args:
        df (pd.DataFrame): DataFrame containing market data.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["price"], label="Price")
    plt.plot(df["time"], df["market_cap"], label="Market Cap")
    plt.plot(df["time"], df["total_volume"], label="Volume")
    plt.title("Market Trends Over Time")
    plt.xlabel("Time")
    plt.ylabel("Normalized Values")
    plt.legend()
    plt.show()


def plot_rolling_statistics(df):
    """
    Plot rolling statistics like moving averages and volatility.

    Args:
        df (pd.DataFrame): DataFrame containing market data.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["price"], label="Price", alpha=0.6)
    plt.plot(df["time"], df["price_7_day_ma"], label="7-Day MA", linestyle="--")
    plt.plot(df["time"], df["price_30_day_ma"], label="30-Day MA", linestyle="--")
    plt.title("Price with Rolling Averages")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


def preprocess_market_data(df):
    """
    Preprocess the market data by adding new features and scaling.

    Args:
        df (pd.DataFrame): DataFrame containing 'price', 'market_cap', and 'total_volume'.

    Returns:
        pd.DataFrame, MinMaxScaler: Processed DataFrame with additional features and the scaler object.
    """
    # Ensure time is a datetime object
    df["time"] = pd.to_datetime(df["time"])

    # Add percentage change features
    df["price_return"] = df["price"].pct_change()
    df["volume_change"] = df["total_volume"].pct_change()
    df["market_cap_change"] = df["market_cap"].pct_change()

    # Add rolling statistics
    df["price_7_day_ma"] = df["price"].rolling(window=7).mean()
    df["price_30_day_ma"] = df["price"].rolling(window=30).mean()
    df["volatility_7_day"] = df["price_return"].rolling(window=7).std()

    # Add ratio features
    df["market_cap_to_volume"] = df["market_cap"] / df["total_volume"]

    # Drop NaN rows introduced by rolling calculations
    df = df.dropna()

    # Scale the features
    features = [
        "price",
        "total_volume",
        "market_cap",
        "price_return",
        "volume_change",
        "market_cap_change",
        "volatility_7_day",
    ]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    return df, scaler


def create_sequences(data, seq_length):
    """
    Create sequences for time series data to be used in LSTM models.

    Args:
        data (np.array): Array of feature data.
        seq_length (int): Number of time steps in each sequence.

    Returns:
        np.array, np.array: Sequences (X) and target values (y).
    """

    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
        targets.append(data[i + seq_length, 0])  # Assuming 'price' is the target
    return np.array(sequences), np.array(targets)


def prepare_lstm_data(df, seq_length):
    """
    Prepare data for LSTM model training and testing.

    Args:
        df (pd.DataFrame): DataFrame containing market data.
        seq_length (int): Number of time steps in each sequence.

    Returns:
        np.array, np.array, np.array, np.array: X_train, X_test, y_train, y_test.
    """
    logging.info("Preparing data for LSTM model...")

    # Split into features and target
    features = df[
        [
            "price",
            "total_volume",
            "market_cap",
            "price_return",
            "volume_change",
            "market_cap_change",
            "volatility_7_day",
        ]
    ].values

    # Create sequences
    X, y = create_sequences(features, seq_length)

    # Split into train and test sets (80/20 split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logging.info("Data preparation for LSTM complete.")
    return X_train, X_test, y_train, y_test


def build_lstm_model(input_shape):
    """
    Build an LSTM model.

    Args:
        input_shape (tuple): Shape of input data.

    Returns:
        keras.Model: Compiled LSTM model.
    """
    logging.info("Building LSTM model...")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    logging.info("LSTM model built and compiled.")
    return model


def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate the LSTM model's performance on test data.

    Args:
        model: Trained LSTM model.
        X_test (np.array): Test data sequences.
        y_test (np.array): True target values.
        scaler (MinMaxScaler): Scaler used for data normalization.

    Returns:
        None
    """
    logging.info("Evaluating LSTM model...")
    # Predict on test data
    predictions = model.predict(X_test)

    # Extract the scaler's min and scale for the target feature (price)
    price_min = scaler.min_[0]
    price_scale = scaler.scale_[0]

    # Rescale predictions and true values to their original scale
    predictions_rescaled = predictions * price_scale + price_min
    y_test_rescaled = y_test.reshape(-1, 1) * price_scale + price_min

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    logging.info(f"Model evaluation complete. MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Plot predictions vs true values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label="True Values")
    plt.plot(predictions_rescaled, label="Predictions")
    plt.title("LSTM Model Predictions vs True Values")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def plot_market_trends(df):
    """
    Plot trends for price, market capitalization, and volume.

    Args:
        df (pd.DataFrame): DataFrame containing market data.
    """
    logging.info("Plotting market trends...")
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["price"], label="Price")
    plt.plot(df["time"], df["market_cap"], label="Market Cap")
    plt.plot(df["time"], df["total_volume"], label="Volume")
    plt.title("Market Trends Over Time")
    plt.xlabel("Time")
    plt.ylabel("Normalized Values")
    plt.legend()
    plt.show()


def plot_rolling_statistics(df):
    """
    Plot rolling statistics like moving averages and volatility.

    Args:
        df (pd.DataFrame): DataFrame containing market data.
    """
    logging.info("Plotting rolling statistics...")
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["price"], label="Price", alpha=0.6)
    plt.plot(df["time"], df["price_7_day_ma"], label="7-Day MA", linestyle="--")
    plt.plot(df["time"], df["price_30_day_ma"], label="30-Day MA", linestyle="--")
    plt.title("Price with Rolling Averages")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("src/streamlit_app/data/market_metrics_data.csv")

    # Preprocess data and generate features
    df, scaler = preprocess_market_data(df)

    # Plot trends and rolling statistics
    plot_market_trends(df)
    plot_rolling_statistics(df)

    # Prepare data for LSTM
    seq_length = 30
    X_train, X_test, y_train, y_test = prepare_lstm_data(df, seq_length)

    # Build and train LSTM model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    # Evaluate model performance
    evaluate_model(model, X_test, y_test, scaler)
