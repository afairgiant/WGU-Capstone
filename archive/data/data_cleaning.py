import os
import logging
import pandas as pd
import yaml


def setup_logging():
    """Sets up logging with a defined format and INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


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
    Perform basic preprocessing on raw cryptocurrency data.

    Args:
        input_file (str): Path to the raw data file.
        output_file (str): Path to save the cleaned data.
    """
    try:
        # Load raw data
        data = pd.read_csv(input_file)

        # Ensure timestamp is in datetime format
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

        # Remove duplicates
        data = data.drop_duplicates()

        # Handle missing timestamps by dropping rows
        data = data.dropna(subset=["timestamp"])

        # Save the cleaned data
        data.to_csv(output_file, index=False)
        print(f"Basic preprocessing complete. Cleaned data saved to {output_file}")
    except Exception as e:
        print(f"Error in basic preprocessing: {e}")
        raise

def preprocess_historical_data(input_file, output_file):
    """
    Perform advanced preprocessing on cleaned cryptocurrency data.

    Args:
        input_file (str): Path to the cleaned data file.
        output_file (str): Path to save the fully preprocessed data.

    Returns:
        pd.DataFrame: The fully preprocessed data as a DataFrame.
    """
    try:
        # Load cleaned data
        data = pd.read_csv(input_file)

        # Add feature engineering
        data["7d_ma"] = data["price"].rolling(window=7).mean()
        data["30d_ma"] = data["price"].rolling(window=30).mean()
        data["price_change"] = data["price"].pct_change() * 100

        # Drop rows with NaN values introduced by rolling windows
        data = data.dropna()

        # Save the preprocessed data
        data.to_csv(output_file, index=False)
        print(f"Advanced preprocessing complete. Preprocessed data saved to {output_file}")

        return data
    except Exception as e:
        print(f"Error in advanced preprocessing: {e}")
        raise

def preprocess_pipeline(raw_input_file, intermediate_output_file, final_output_file):
    """
    Combine basic and advanced preprocessing steps into a pipeline.

    Args:
        raw_input_file (str): Path to the raw input data.
        intermediate_output_file (str): Path to save basic cleaned data.
        final_output_file (str): Path to save fully preprocessed data.

    Returns:
        pd.DataFrame: The final preprocessed data as a DataFrame.
    """
    print("Starting preprocessing pipeline...")

    # Step 1: Basic preprocessing
    preprocess_data(raw_input_file, intermediate_output_file)

    # Step 2: Advanced preprocessing
    final_data = preprocess_historical_data(intermediate_output_file, final_output_file)

    print("Preprocessing pipeline complete.")
    return final_data

def process_data(config_file):
    """
    Orchestrates the preprocessing pipeline for all cryptocurrencies.

    Args:
        config_file (str): Path to the configuration file.
    """
    logging.info("Loading configuration...")
    config = load_config(config_file)

    # Extract parameters
    cryptocurrencies = config["cryptocurrencies"]
    raw_dir = "data/raw"
    intermediate_dir = "data/intermediate"
    processed_dir = "data/processed"

    # Ensure required directories exist
    for directory in [raw_dir, intermediate_dir, processed_dir]:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Directory ensured: {directory}")

    # Process each cryptocurrency
    for crypto_id in cryptocurrencies:
        raw_file = os.path.join(raw_dir, f"{crypto_id}_historical.csv")
        intermediate_file = os.path.join(intermediate_dir, f"{crypto_id}_cleaned.csv")
        final_file = os.path.join(processed_dir, f"{crypto_id}_final.csv")

        if not os.path.exists(raw_file):
            logging.warning(f"Raw file not found for {crypto_id}: {raw_file}")
            continue

        try:
            preprocess_pipeline(raw_file, intermediate_file, final_file)
        except Exception as e:
            logging.error(f"Error processing data for {crypto_id}: {e}")
            continue


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
