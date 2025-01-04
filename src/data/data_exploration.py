import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def load_config(config_file):
    try:
        with open(config_file, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def load_cleaned_data(crypto_id, data_dir):
    file_path = os.path.join(data_dir, f"{crypto_id}_cleaned.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading data for {crypto_id}: {e}")


def plot_price_trend(data, crypto_id):
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(
            data["timestamp"], data["price"], label=f"{crypto_id.capitalize()} Price"
        )
        plt.title(f"{crypto_id.capitalize()} Price Trend")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        raise ValueError(f"Error plotting price trend for {crypto_id}: {e}")


def plot_price_distribution(data, crypto_id):
    try:
        plt.figure(figsize=(8, 5))
        sns.histplot(data["price"], bins=30, kde=True)
        plt.title(f"{crypto_id.capitalize()} Price Distribution")
        plt.xlabel("Price (USD)")
        plt.ylabel("Frequency")
        plt.show()
    except Exception as e:
        raise ValueError(f"Error plotting price distribution for {crypto_id}: {e}")


if __name__ == "__main__":
    config_file = "configs/config.yaml"
    try:
        config = load_config(config_file)
    except Exception as e:
        raise SystemExit(f"Failed to load configuration: {e}")

    cryptocurrencies = config.get("cryptocurrencies", [])
    data_dir = "data/processed"

    if not cryptocurrencies:
        raise ValueError("No cryptocurrencies specified in the configuration.")

    for crypto_id in cryptocurrencies:
        print(f"Exploring data for {crypto_id}...")
        try:
            data = load_cleaned_data(crypto_id, data_dir)
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            plot_price_trend(data, crypto_id)
            plot_price_distribution(data, crypto_id)
        except Exception as e:
            print(f"Error processing data for {crypto_id}: {e}")

    print("Data exploration completed!")
