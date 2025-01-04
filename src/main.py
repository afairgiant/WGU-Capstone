import logging
import os

import yaml

from src.data.data_cleaning import preprocess_data
from src.data.data_loader import fetch_historical_data


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
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


def create_directories(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Directory ensured: {directory}")


def process_data(config_file="configs/config.yaml"):
    setup_logging()

    config = load_config(config_file)
    cryptocurrencies = config.get("cryptocurrencies", [])
    vs_currency = config.get("vs_currency", "usd")
    days = config.get("days", 30)

    if not cryptocurrencies:
        logging.error("No cryptocurrencies specified in the configuration.")
        raise ValueError("Cryptocurrencies must be specified in the configuration.")

    raw_dir = "data/raw"
    processed_dir = "data/processed"

    create_directories([raw_dir, processed_dir])

    logging.info("Fetching raw data...")
    fetch_historical_data(cryptocurrencies, vs_currency, days, raw_dir)

    logging.info("Cleaning data...")
    for crypto_id in cryptocurrencies:
        raw_file = os.path.join(raw_dir, f"{crypto_id}_historical.csv")
        processed_file = os.path.join(processed_dir, f"{crypto_id}_cleaned.csv")

        if not os.path.exists(raw_file):
            logging.warning(f"Raw file not found for {crypto_id}: {raw_file}")
            continue

        preprocess_data(raw_file, processed_file)
        logging.info(f"Processed {crypto_id} data: {processed_file}")


if __name__ == "__main__":
    try:
        process_data()
    except Exception as e:
        logging.error(f"Critical error in processing: {e}")
