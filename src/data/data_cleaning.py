import os

import pandas as pd
import yaml


def load_config(config_file):
    """
    Load configuration from a YAML file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


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


if __name__ == "__main__":
    # Load configuration
    config_file = "configs/config.yaml"
    config = load_config(config_file)

    # Extract parameters
    cryptocurrencies = config["cryptocurrencies"]
    output_dir_raw = "data/raw"
    output_dir_processed = "data/processed"

    # Ensure the output directory exists
    os.makedirs(output_dir_processed, exist_ok=True)

    # Process each cryptocurrency's data
    for crypto_id in cryptocurrencies:
        input_file = os.path.join(output_dir_raw, f"{crypto_id}_historical.csv")
        output_file = os.path.join(output_dir_processed, f"{crypto_id}_cleaned.csv")
        preprocess_data(input_file, output_file)
