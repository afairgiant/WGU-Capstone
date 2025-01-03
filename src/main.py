import os

from src.data.data_cleaning import preprocess_data
from src.data.data_loader import fetch_historical_data


def main():
    # Set parameters
    crypto_id = "bitcoin"
    vs_currency = "usd"
    days = 30
    raw_file = os.path.join("data", "raw", f"{crypto_id}_historical.csv")
    processed_file = os.path.join("data", "processed", f"{crypto_id}_cleaned.csv")

    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Step 1: Fetch raw data
    print("Fetching raw data...")
    fetch_historical_data(crypto_id, vs_currency, days, raw_file)

    # Step 2: Clean and preprocess data
    print("Cleaning data...")
    preprocess_data(raw_file, processed_file)

    print("Process completed!")


if __name__ == "__main__":
    main()
