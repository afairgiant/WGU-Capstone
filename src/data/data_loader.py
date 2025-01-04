import json
import logging
import os

import pandas as pd
import requests
import yaml

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


def fetch_historical_data(crypto_ids, vs_currency, days, output_path):
    for crypto_id in crypto_ids:
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": days}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            prices = pd.DataFrame(
                data.get("prices", []), columns=["timestamp", "price"]
            )
            if prices.empty:
                logging.warning(f"No price data available for {crypto_id}")
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
