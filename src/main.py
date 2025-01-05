import logging
import os
import sys

import pandas as pd
import yaml

# Dynamically add the project root to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import data_exploration
from src.data.data_cleaning import process_data
from src.data.data_exploration import explore_data, predict_future
from src.data.data_loader import fetch_historical_data
from src.utils.utils import (
    create_directories,
    load_config,
    logger,
    setup_logging,
    validate_config,
)


def download_raw_data(config_file):
    """
    Downloads raw cryptocurrency data using data_loader.py.

    Args:
        config_file (str): Path to the configuration file.
    """
    logging.info("Starting raw data download...")

    # Load configuration
    config = load_config(config_file)

    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        logging.error(f"Configuration validation error: {e}")
        raise

    cryptocurrencies = config["cryptocurrencies"]
    vs_currency = config.get("vs_currency", "usd")
    days = config.get("days", 90)
    raw_dir = "data/raw"

    # Ensure raw data directory exists
    create_directories([raw_dir])

    try:
        fetch_historical_data(cryptocurrencies, vs_currency, days, raw_dir)
        logging.info("Raw data download complete.")
    except Exception as e:
        logging.error(f"Error downloading raw data: {e}")
        raise


def call_visuals(config_file=None):
    logging.info("Exploring data...")
    # Determine the configuration file path
    config_file = config_file or os.getenv("CONFIG_FILE", "configs/config.yaml")

    config = load_config(config_file)

    try:
        for crypto_id in config["cryptocurrencies"]:
            processed_file = os.path.join("data/processed", f"{crypto_id}_final.csv")
            if not os.path.exists(processed_file):
                logging.warning(
                    f"Processed file not found for {crypto_id}: {processed_file}"
                )
                continue
            try:
                # Load data as a DataFrame
                data = pd.read_csv(processed_file)

                # Ensure timestamp column is in the correct format
                data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
                data = data.dropna(
                    subset=["timestamp"]
                )  # Remove rows with invalid timestamps

                # Call explore_data with the DataFrame
                explore_data(crypto_id, data)
                predict_future(crypto_id, data, 30)

            except Exception as e:
                logging.error(f"Error exploring data for {crypto_id}: {e}")
                continue

    except Exception as e:
        logging.error(f"Error exploring data: {e}")
        raise


def main():
    """
    Main entry point of the program.
    Handles raw data download, preprocessing, and optional exploration.
    """
    setup_logging()
    logging.info("Starting main program...")

    config_file = "configs/config.yaml"

    try:
        # Step 1: Download raw data
        download_raw_data(config_file)

        # Step 2: Process data (cleaning and feature engineering)
        process_data(config_file)

        # Step 3: (Optional) Explore processed data
        call_visuals(config_file)

        logging.info("Program completed successfully.")
    except Exception as e:
        logging.critical(f"Critical error in main program: {e}")
        raise


if __name__ == "__main__":
    main()
