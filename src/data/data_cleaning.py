import os

import pandas as pd


def preprocess_data(input_file, output_file):
    """
    Preprocess raw cryptocurrency data.

    Args:
        input_file (str): Path to the raw data file.
        output_file (str): Path to save the cleaned data.
    """
    # Load raw data
    data = pd.read_csv(input_file)

    # Ensure timestamp is in datetime format
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Remove duplicates
    data = data.drop_duplicates()

    # Save the cleaned data
    data.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")


# if __name__ == "__main__":
#     raw_file = os.path.join("data", "raw", "dogecoin_historical.csv")
#     processed_file = os.path.join("data", "processed", "dogecoin_cleaned.csv")

#     preprocess_data(raw_file, processed_file)
