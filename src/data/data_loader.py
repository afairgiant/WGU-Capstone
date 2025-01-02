import os
from datetime import datetime

import pandas as pd
import requests


def fetch_historical_data(crypto_id, vs_currency, days):
    """
    Fetch historical market data for a cryptocurrency from the CoinGecko API.

    Args:
        crypto_id (str): The CoinGecko ID for the cryptocurrency (e.g., 'bitcoin').
        vs_currency (str): The currency to compare against (e.g., 'usd').
        days (int): Number of past days to fetch data for.

    Returns:
        pd.DataFrame: Historical data as a DataFrame.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
        return prices
    else:
        print(f"Error fetching data: {response.status_code}")
        return None


if __name__ == "__main__":
    # Example usage
    crypto_id = "bitcoin"  # ID for Bitcoin
    vs_currency = "usd"  # Comparing against USD
    days = 30  # Fetch data for the past 30 days

    data = fetch_historical_data(crypto_id, vs_currency, days)

    if data is not None:
        output_path = os.path.join("data", "raw", f"{crypto_id}_historical.csv")
        data.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
