import os
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from matplotlib.pylab import f
from pyparsing import C
from utils import load_api_key


# Utility function for API requests
def fetch_data_from_api(url, params=None, headers=None):
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")


# Utility function to save DataFrame to CSV
def save_dataframe_to_csv(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Data saved successfully to {file_path}")


# Utility function to load or create a DataFrame
def load_or_create_dataframe(file_path, columns):
    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=["time"])
    return pd.DataFrame(columns=columns)


# Utility function to update metadata
def update_metadata(file_path, timestamp=None):
    timestamp = timestamp or datetime.utcnow()
    with open(file_path, "w") as file:
        file.write(timestamp.strftime("%Y-%m-%d %H:%M:%S"))


# Simplified OHLC data downloader
def download_ohlc_data(
    coin_id,
    days,
    output_file="src/streamlit_app/data/ohlc_data.csv",
    metadata_file="src/streamlit_app/data/ohlc_last_download.txt",
):
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as file:
            last_download = datetime.strptime(file.read().strip(), "%Y-%m-%d %H:%M:%S")
            if datetime.utcnow() - last_download < timedelta(hours=6):
                print("Data recently fetched. Skipping download.")
                return output_file

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
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
    api_key_path,
    key_name,
    output_file,
    metadata_file="src/streamlit_app/data/market_last_download.txt",
):
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as file:
            last_download = datetime.strptime(file.read().strip(), "%Y-%m-%d %H:%M:%S")
            if datetime.utcnow() - last_download < timedelta(hours=6):
                print("Data recently fetched. Skipping download.")
                return output_file

    api_key = load_api_key(api_key_path, key_name)
    if not api_key:
        raise ValueError("Missing API key. Check your configuration.")

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
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
    # return df
    return output_file


# Example usage
if __name__ == "__main__":
    # Parameters
    COIN_ID = "bitcoin"
    DAYS = 30

    OUTPUT_FILE = "src/streamlit_app/data/ohlc_data.csv"
    OUTPUT_FILE_2 = "src/streamlit_app/data/ohlc_data_training.csv"
    OUTPUT_FILE_3 = "src/streamlit_app/data/market_metrics_data.csv"
    # Download, clean, and save data
    # download_and_save_ohlc_data(COIN_ID, DAYS, OUTPUT_FILE)
    # download_and_append_ohlc_data(COIN_ID, OUTPUT_FILE_2)
    # fetch_max_ohlc_data(COIN_ID)
    bitcoin_metrics = download_blockchain_metrics(
        "configs/api_keys.json", "apiKey_gecko", OUTPUT_FILE_3
    )
