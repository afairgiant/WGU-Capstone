import os
import time

import pandas as pd
import requests

API_KEY = "CG-jMJRBmwSsWizV5LXEExWkn9K"


def download_and_save_ohlc_data(
    coin_id, days, output_file="src/streamlit_app/data/ohlc_data.csv"
):
    """
    Downloads OHLC data for a cryptocurrency from the CoinGecko API, cleans it, and saves it to a CSV file.

    Args:
        coin_id (str): The cryptocurrency ID (e.g., 'bitcoin').
        days (int): Number of days of historical data to retrieve.
        output_file (str): The path to the CSV file where data will be saved.

    Returns:
        str: The path to the saved CSV file.
    """
    try:
        # Fetch OHLC data from CoinGecko
        response = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
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

        # Save the cleaned data to a CSV file
        df.to_csv(output_file, index=False)

        print(f"Data saved successfully to {output_file}")
        return output_file

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

    # Download, clean, and save data
    download_and_save_ohlc_data(COIN_ID, DAYS, OUTPUT_FILE)
