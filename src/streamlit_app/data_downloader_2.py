import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from pyparsing import C
from utils import load_api_key


def download_and_save_ohlc_data(
    coin_id,
    days,
    output_file="src/streamlit_app/data/ohlc_data.csv",
    metadata_file="src/streamlit_app/data/ohlc_last_download.txt",
):
    """
    Downloads OHLC data for a cryptocurrency from the CoinGecko API, appends missing data to the CSV,
    and ensures only the last 'days' days are retained. Skips download if last fetch was within 6 hours.

    Args:
        coin_id (str): The cryptocurrency ID (e.g., 'bitcoin').
        days (int): Number of days of historical data to retrieve.
        output_file (str): The path to the CSV file where data will be saved.
        metadata_file (str): Path to the file storing the last download timestamp.

    Returns:
        str: The path to the updated CSV file.
    """
    try:
        # Check the last download time
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as file:
                last_download_str = file.read().strip()
            last_download_time = datetime.strptime(
                last_download_str, "%Y-%m-%d %H:%M:%S"
            )
            elapsed_time = datetime.utcnow() - last_download_time

            if elapsed_time < timedelta(hours=6):
                print(f"Data was last downloaded {elapsed_time} ago. Skipping fetch.")
                return output_file

        # Determine the date range for the past 'days' days
        today = datetime.utcnow()
        start_date = today - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=today)

        # Load existing data if the file exists
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file, parse_dates=["time"])
        else:
            existing_df = pd.DataFrame(columns=["time", "open", "high", "low", "close"])

        # Identify missing dates
        existing_dates = (
            existing_df["time"].dt.normalize().unique() if not existing_df.empty else []
        )
        missing_dates = [date for date in date_range if date not in existing_dates]

        # Download and append missing data
        if missing_dates:
            print(f"Fetching data for {len(missing_dates)} missing days...")
            response = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
                params={"vs_currency": "usd", "days": days},
            )
            response.raise_for_status()
            ohlc_data = response.json()

            # Convert the data into a DataFrame
            new_df = pd.DataFrame(
                ohlc_data, columns=["time", "open", "high", "low", "close"]
            )
            new_df["time"] = pd.to_datetime(new_df["time"], unit="ms")

            # Append and deduplicate data
            updated_df = (
                pd.concat([existing_df, new_df])
                .drop_duplicates(subset=["time"])
                .sort_values("time")
            )

            # Save the updated data to the CSV file
            updated_df.to_csv(output_file, index=False)
            print(f"Data updated and saved successfully to {output_file}")

            # Update the last download timestamp
            with open(metadata_file, "w") as file:
                file.write(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        else:
            print("No missing data to fetch. CSV file is already up-to-date.")

        return output_file

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        raise

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def download_and_append_ohlc_data(
    coin_id, output_file="src/streamlit_app/data/ohlc_data_2.csv"
):
    """
    Downloads the latest 24 hours of OHLC data for a cryptocurrency from the CoinGecko API,
    appends it to an existing CSV file, and removes duplicates.

    Args:
        coin_id (str): The cryptocurrency ID (e.g., 'bitcoin').
        output_file (str): The path to the CSV file where data will be saved.

    Returns:
        str: The path to the saved CSV file.
    """
    try:
        # Fetch the latest 24-hour OHLC data from CoinGecko
        response = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=1"
        )
        response.raise_for_status()
        ohlc_data = response.json()

        # Ensure data is not empty
        if not ohlc_data:
            raise ValueError("No data retrieved from the API.")

        # Convert the data into a DataFrame
        df = pd.DataFrame(ohlc_data, columns=["time", "open", "high", "low", "close"])

        # Convert the timestamp to a readable datetime format
        df["time"] = pd.to_datetime(df["time"], unit="ms")

        # Check if the file exists
        if os.path.exists(output_file):
            # Load existing data
            existing_df = pd.read_csv(output_file)

            # Convert time column to datetime for consistency
            existing_df["time"] = pd.to_datetime(existing_df["time"])

            # Append the new data to the existing data
            combined_df = pd.concat([existing_df, df])

            # Drop duplicate rows based on the "time" column
            combined_df = combined_df.drop_duplicates(subset="time").sort_values(
                by="time"
            )
            print("Data appended successfully.")

            # Save the combined DataFrame back to the CSV
            combined_df.to_csv(output_file, index=False)
        else:
            # Save new data directly if the file doesn't exist
            df.to_csv(output_file, index=False)

        print(f"Data saved successfully to {output_file}")
        return output_file

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        raise

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def fetch_max_ohlc_data(
    coin_id,
    currency="usd",
    output_file="historical_ohlc_data.csv",
    api_key_path="configs/api_keys.json",
    key_name="apiKey_gecko",
):
    """
    Fetches the maximum OHLC data allowed by CoinGecko's demo API (past 365 days).

    Args:
        coin_id (str): The cryptocurrency ID (e.g., 'bitcoin').
        currency (str): The fiat currency (default is 'usd').
        output_file (str): The path to save the OHLC data as a CSV.
        api_key_path (str, optional): Path to the API key file (if using a Pro-API key).
        key_name (str, optional): Key name for loading the API key.

    Returns:
        pd.DataFrame: A DataFrame containing the historical OHLC data.
    """
    try:
        # Calculate the maximum range for the demo API
        today = datetime.utcnow()
        start_date = today - timedelta(days=365)
        days = 365

        # Load API key if provided
        headers = {}
        if api_key_path and key_name:
            api_key = load_api_key(api_key_path, key_name)
            headers["Authorization"] = f"Bearer {api_key}"

        # Fetch OHLC data from CoinGecko API
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        params = {
            "vs_currency": currency,
            "days": days,
        }
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        # Process API response
        ohlc_data = response.json()
        if not ohlc_data:
            raise ValueError("No OHLC data returned by the API.")

        # Convert the data into a DataFrame
        df = pd.DataFrame(
            ohlc_data, columns=["timestamp", "open", "high", "low", "close"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Save the DataFrame to a CSV file
        df.to_csv(output_file, index=False)
        print(f"OHLC data saved successfully to {output_file}")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        raise

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


# Example usage
if __name__ == "__main__":
    # Parameters
    COIN_ID = "bitcoin"
    DAYS = 30

    OUTPUT_FILE = "src/streamlit_app/data/ohlc_data.csv"
    OUTPUT_FILE_2 = "src/streamlit_app/data/ohlc_data_training.csv"

    # Download, clean, and save data
    # download_and_save_ohlc_data(COIN_ID, DAYS, OUTPUT_FILE)
    # download_and_append_ohlc_data(COIN_ID, OUTPUT_FILE_2)
    fetch_max_ohlc_data(COIN_ID)
