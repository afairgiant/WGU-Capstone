import logging
import os
import sys

import yaml

# Dynamically add the project root to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import data_exploration
from src.data.data_cleaning import preprocess_data
from src.data.data_exploration import explore_data
from src.data.data_loader import fetch_historical_data


def setup_logging():
    """Sets up logging with a defined format and DEBUG level."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_config(config_file):
    """Loads configuration from a YAML file."""
    try:
        with open(config_file, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        raise


def validate_config(config):
    """Validates the required configuration values."""
    if not config.get("cryptocurrencies"):
        raise ValueError("Cryptocurrencies must be specified in the configuration.")
    if config.get("days", 0) <= 0:
        raise ValueError("Days must be a positive integer.")


def create_directories(directories):
    """Ensures specified directories exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Directory ensured: {directory}")


def process_data(config_file=None):
    """Main function to process cryptocurrency data."""
    setup_logging()

    # Determine the configuration file path
    config_file = config_file or os.getenv("CONFIG_FILE", "configs/config.yaml")

    logging.info(f"Loading configuration from: {config_file}")
    config = load_config(config_file)

    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        logging.error(f"Configuration validation error: {e}")
        raise

    cryptocurrencies = config["cryptocurrencies"]
    vs_currency = config.get("vs_currency", "usd")
    days = config.get("days", 30)

    raw_dir = "data/raw"
    processed_dir = "data/processed"

    create_directories([raw_dir, processed_dir])

    logging.info("Fetching raw data...")
    try:
        fetch_historical_data(cryptocurrencies, vs_currency, days, raw_dir)
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        raise

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
            logging.error(f"Error processing data for {crypto_id}: {e}")
            continue


def visuals(config_file=None):
    logging.info("Exploring data...")
    # Determine the configuration file path
    config_file = config_file or os.getenv("CONFIG_FILE", "configs/config.yaml")

    config = load_config(config_file)

    try:
        for crypto_id in config["cryptocurrencies"]:
            processed_file = os.path.join("data/processed", f"{crypto_id}_cleaned.csv")
            if not os.path.exists(processed_file):
                logging.warning(
                    f"Processed file not found for {crypto_id}: {processed_file}"
                )
                continue
            try:
                explore_data(crypto_id, processed_file)
            except Exception as e:
                logging.error(f"Error exploring data for {crypto_id}: {e}")
                continue
    except Exception as e:
        logging.error(f"Error exploring data: {e}")
        raise


def main():
    process_data()
    visuals()


if __name__ == "__main__":
    try:
        process_data()
        visuals()
    except Exception as e:
        logging.critical(f"Critical error in processing: {e}")
