import json
import logging
import os

import pandas as pd
import requests
import yaml

API_KEY = "CG-jMJRBmwSsWizV5LXEExWkn9K "

# Ensure the logs directory exists
log_dir = os.path.join(os.path.dirname(__file__), "../../logs")
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, "data_loader.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_config(config_file):
    try:
        with open(config_file, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        raise


def save_metadata(crypto_id, output_path):
    metadata = {
        "crypto_id": crypto_id,
        "source": "CoinGecko",
        "fetch_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(os.path.join(output_path, f"{crypto_id}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Metadata for {crypto_id} saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving metadata for {crypto_id}: {e}")
        raise


def get_coin_list():
    """
    Fetch the list of available coins from the CoinGecko API and save it to a YAML file.
    """
    url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&api_key={API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        coins_list = response.json()
        coins = [coin["id"] for coin in coins_list]

        # Save the list of coins to a YAML file
        output_path = "configs/coins.yaml"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(coins, f)
        logging.info(f"Coin list saved to {output_path}")
    except requests.RequestException as e:
        logging.error(f"HTTP error fetching coin list: {e}")
    except Exception as e:
        logging.error(f"Unexpected error fetching coin list: {e}")


def fetch_historical_data(crypto_ids, vs_currency, days, output_path):
    """
    Fetch historical price data for given cryptocurrencies and save to CSV files.

    Args:
        crypto_ids (list): List of cryptocurrency IDs (e.g., ['bitcoin', 'ethereum']).
        vs_currency (str): The target currency (e.g., 'usd').
        days (int): Number of days of historical data to fetch.
        output_path (str): Directory to save the fetched data.
    """
    for crypto_id in crypto_ids:
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": days}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            prices = pd.DataFrame(
                data.get("prices", []), columns=["timestamp", "price"]
            )
            if prices.empty:
                logging.warning(f"No price data available for %s", crypto_id)
                continue

            prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
            output_file = os.path.join(output_path, f"{crypto_id}_historical.csv")
            prices.to_csv(output_file, index=False)
            logging.info(f"Data for {crypto_id} saved to {output_file}")

            save_metadata(crypto_id, output_path)
        except requests.RequestException as e:
            logging.error(f"HTTP error fetching data for {crypto_id}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error fetching data for {crypto_id}: {e}")


if __name__ == "__main__":
    try:
        get_coin_list()
        config = load_config("configs/config.yaml")

        crypto_ids = config.get("cryptocurrencies", [])
        vs_currency = config.get("vs_currency", "usd")
        days = config.get("days", 30)
        output_dir = "data/raw"

        if not crypto_ids:
            logging.error("No cryptocurrencies specified in the configuration.")
            raise ValueError(
                "The configuration must specify at least one cryptocurrency."
            )

        os.makedirs(output_dir, exist_ok=True)
        fetch_historical_data(crypto_ids, vs_currency, days, output_dir)
    except Exception as e:
        logging.critical(f"Critical error in data loading process: {e}")
