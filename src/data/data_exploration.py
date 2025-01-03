import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def load_config(config_file):
    """
    Load configuration from a YAML file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def load_cleaned_data(crypto_id, data_dir):
    """
    Load the cleaned data for a given cryptocurrency.

    Args:
        crypto_id (str): Cryptocurrency ID (e.g., 'bitcoin').
        data_dir (str): Directory where cleaned data is stored.

    Returns:
        pd.DataFrame: Cleaned data for the cryptocurrency.
    """
    file_path = os.path.join(data_dir, f"{crypto_id}_cleaned.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def plot_price_trend(data, crypto_id):
    """
    Plot the price trend for a given cryptocurrency.

    Args:
        data (pd.DataFrame): DataFrame containing cleaned data.
        crypto_id (str): Cryptocurrency ID.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data["timestamp"], data["price"], label=f"{crypto_id.capitalize()} Price")
    plt.title(f"{crypto_id.capitalize()} Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_price_distribution(data, crypto_id):
    """
    Plot the price distribution for a given cryptocurrency.

    Args:
        data (pd.DataFrame): DataFrame containing cleaned data.
        crypto_id (str): Cryptocurrency ID.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(data["price"], bins=30, kde=True)
    plt.title(f"{crypto_id.capitalize()} Price Distribution")
    plt.xlabel("Price (USD)")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    # Load configuration
    config_file = "configs/config.yaml"
    config = load_config(config_file)

    # Extract parameters
    cryptocurrencies = config["cryptocurrencies"]
    data_dir = "data/processed"

    # Perform data exploration for each cryptocurrency
    for crypto_id in cryptocurrencies:
        print(f"Exploring data for {crypto_id}...")

        # Load the cleaned data
        data = load_cleaned_data(crypto_id, data_dir)

        # Convert timestamp to datetime for plotting
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        # Plot price trends
        plot_price_trend(data, crypto_id)

        # Plot price distribution
        plot_price_distribution(data, crypto_id)

    print("Data exploration completed!")
