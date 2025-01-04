import logging
import os
import sys

import yaml

# Add the project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from src.data.data_cleaning import preprocess_data
from src.data.data_loader import fetch_historical_data


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_file):
    try:
        with open(config_file, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        sys.exit(1)


def create_directories(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Directory ensured: {directory}")


def main():
    setup_logging()

    config_file = "configs/config.yaml"
    config = load_config(config_file)

    cryptocurrencies = config.get("cryptocurrencies", [])
    vs_currency = config.get("vs_currency", "usd")
    days = config.get("days", 30)

    if not cryptocurrencies:
        logging.error("No cryptocurrencies specified in the configuration.")
        sys.exit(1)

    raw_dir = "data/raw"
    processed_dir = "data/processed"

    create_directories([raw_dir, processed_dir])

    logging.info("Fetching raw data...")
    try:
        fetch_historical_data(cryptocurrencies, vs_currency, days, raw_dir)
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        sys.exit(1)

    logging.info("Cleaning data...")
    for crypto_id in cryptocurrencies:
        raw_file = os.path.join(raw_dir, f"{crypto_id}_historical.csv")
        processed_file = os.path.join(processed_dir, f"{crypto_id}_cleaned.csv")

        if not os.path.exists(raw_file):
            logging.warning(f"Raw file not found for {crypto_id}: {raw_file}")
            continue

        try:
            preprocess_data(raw_file, processed_file)
            logging.info(f"Processed {crypto_id} data: {processed_file}")
        except Exception as e:
            logging.error(f"Error processing {crypto_id}: {e}")

    logging.info("Process completed successfully!")


if __name__ == "__main__":
    main()
