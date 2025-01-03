import logging
import os
from datetime import datetime

import pandas as pd
import requests
import yaml

# Configure logging
logging.basicConfig(
    filename="logs/data_loader.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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


def fetch_historical_data(crypto_ids, vs_currency, days, output_path):
    """
    Fetch historical market data for multiple cryptocurrencies from the CoinGecko API.

    Args:
        crypto_ids (list): List of CoinGecko IDs for cryptocurrencies (e.g., ['bitcoin', 'ethereum']).
        vs_currency (str): The currency to compare against (e.g., 'usd').
        days (int): Number of past days to fetch data for.
        output_dir (str): Directory to save the output CSV files.

    Returns:
        None
    """
    for crypto_id in crypto_ids:
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": days}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
            prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
            output_path = os.path.join(output_dir, f"{crypto_id}_historical.csv")
            prices.to_csv(output_path, index=False)
            print(f"Data for {crypto_id} saved to {output_path}")

        else:
            print(f"Error fetching data for {crypto_id}: {response.status_code}")


if __name__ == "__main__":
    # Load configuration
    config = load_config("configs/config.yaml")

    # Extract parameters
    crypto_ids = config["cryptocurrencies"]
    vs_currency = config["vs_currency"]
    days = config["days"]
    output_dir = "data/raw"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Fetch data using parameters from the configuration file
    fetch_historical_data(crypto_ids, vs_currency, days, output_dir)
