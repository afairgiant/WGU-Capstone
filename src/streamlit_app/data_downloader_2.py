import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from utils import load_api_key

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
API_BASE_URL = "https://api.coingecko.com/api/v3"
DATA_DIR = "src/streamlit_app/data"
DEFAULT_OHLC_OUTPUT_FILE = os.path.join(DATA_DIR, "ohlc_data.csv")
DEFAULT_OHLC_METADATA_FILE = os.path.join(DATA_DIR, "ohlc_last_download.txt")
DEFAULT_MARKET_METAADATA_FILE = os.path.join(DATA_DIR, "market_last_download.txt")
DEFAULT_MARKET_OUTPUT_FILE = os.path.join(DATA_DIR, "market_metrics_data.csv")


# Utility function for API requests
def fetch_data_from_api(
    url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None
) -> Union[Dict, List]:
    """
    Fetch data from an API with error handling.

    Args:
        url (str): API endpoint URL.
        params (dict, optional): Query parameters for the API.
        headers (dict, optional): Headers for the API request.

    Returns:
        dict: Parsed JSON response from the API.
    """
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        raise RuntimeError(f"API request failed: {e}")


# Utility function to save DataFrame to CSV
def save_dataframe_to_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Path to the output file.
    """
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")
        raise


# Utility function to load or create a DataFrame
def load_or_create_dataframe(file_path: str, columns: List[str]) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file or create an empty one if the file doesn't exist.

    Args:
        file_path (str): Path to the CSV file.
        columns (list): Column names for the new DataFrame if the file doesn't exist.

    Returns:
        pd.DataFrame: Loaded or newly created DataFrame.
    """
    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=["time"])
    return pd.DataFrame(columns=columns)


# Utility function to update metadata
def update_metadata(file_path: str, timestamp: Optional[datetime] = None) -> None:
    """
    Update the metadata file with the last download timestamp.

    Args:
        file_path (str): Path to the metadata file.
        timestamp (datetime, optional): Timestamp to save. Defaults to the current UTC time.
    """
    timestamp = timestamp or datetime.utcnow()
    try:
        with open(file_path, "w") as file:
            file.write(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        logging.info(f"Metadata updated at {file_path}")
    except Exception as e:
        logging.error(f"Error updating metadata: {e}")
        raise


# OHLC data downloader
def download_ohlc_data(
    coin_id: str,
    days: int,
    output_file: str = DEFAULT_OHLC_OUTPUT_FILE,
    metadata_file: str = DEFAULT_OHLC_METADATA_FILE,
):
    """
    Download OHLC (Open-High-Low-Close) data for a given cryptocurrency.

    Args:
        coin_id (str): Coin ID (e.g., 'bitcoin').
        days (int): Number of past days to fetch.
        output_file (str, optional): Path to the output CSV file. Defaults to DEFAULT_OUTPUT_FILE.
        metadata_file (str, optional): Path to the metadata file. Defaults to DEFAULT_METADATA_FILE.

    Returns:
        str: Path to the saved CSV file.
    """
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as file:
            last_download = datetime.strptime(file.read().strip(), "%Y-%m-%d %H:%M:%S")
            if datetime.utcnow() - last_download < timedelta(hours=6):
                logging.info("Data recently fetched. Skipping download.")
                return output_file

    url = f"{API_BASE_URL}/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    ohlc_data = fetch_data_from_api(url, params=params)
    if not ohlc_data:
        raise ValueError("No data returned by the API.")

    df = pd.DataFrame(ohlc_data, columns=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")

    existing_df = load_or_create_dataframe(
        output_file, ["time", "open", "high", "low", "close"]
    )
    updated_df = (
        pd.concat([existing_df, df]).drop_duplicates(subset="time").sort_values("time")
    )
    save_dataframe_to_csv(updated_df, output_file)

    update_metadata(metadata_file)
    return output_file


# Download blockchain metrics
def download_blockchain_metrics(
    api_key_path: str,
    key_name: str,
    output_file: str = DEFAULT_MARKET_OUTPUT_FILE,
    metadata_file: str = DEFAULT_MARKET_METAADATA_FILE,
):
    """
    Download blockchain metrics such as prices, market cap, and volume.

    Args:
        api_key_path (str): Path to the API key file.
        key_name (str): Key name in the API key file.
        output_file (str): Path to the output CSV file.
        metadata_file (str, optional): Path to the metadata file. Defaults to DEFAULT_METADATA_FILE.

    Returns:
        str: Path to the saved CSV file.
    """
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as file:
            last_download = datetime.strptime(file.read().strip(), "%Y-%m-%d %H:%M:%S")
            if datetime.utcnow() - last_download < timedelta(hours=6):
                logging.info("Data recently fetched. Skipping download.")
                return output_file

    api_key = load_api_key(api_key_path, key_name)
    if not api_key:
        raise ValueError("Missing API key. Check your configuration.")

    url = f"{API_BASE_URL}/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": 365, "interval": "daily"}
    headers = {"Authorization": f"Bearer {api_key}"}

    data = fetch_data_from_api(url, params=params, headers=headers)
    prices, market_caps, total_volumes = (
        data.get("prices", []),
        data.get("market_caps", []),
        data.get("total_volumes", []),
    )
    if not prices:
        raise ValueError("No data retrieved from the API.")

    df = pd.DataFrame(
        {
            "time": [pd.to_datetime(price[0], unit="ms") for price in prices],
            "price": [price[1] for price in prices],
            "market_cap": [cap[1] for cap in market_caps],
            "total_volume": [vol[1] for vol in total_volumes],
        }
    )
    save_dataframe_to_csv(df, output_file)
    update_metadata(metadata_file)

    return output_file


# Example usage
if __name__ == "__main__":
    # Parameters
    COIN_ID = "bitcoin"
    DAYS = 30

    # Download, clean, and save data
    try:
        download_ohlc_data(COIN_ID, DAYS, DEFAULT_OHLC_OUTPUT_FILE)
        # download_ohlc_data(COIN_ID, DAYS, OHLC_TRAINING_FILE)
        download_blockchain_metrics(
            "configs/api_keys.json", "apiKey_gecko", DEFAULT_MARKET_OUTPUT_FILE
        )
    except Exception as e:
        logging.error(f"An error occurred: {e}")
