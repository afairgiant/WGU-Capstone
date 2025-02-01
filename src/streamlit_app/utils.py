import json
import logging
import os
from logging.handlers import RotatingFileHandler


# Load the API key from the JSON file
def load_api_key(json_file_path, key_name):
    """
    Load the API key from a JSON file.

    Args:
        json_file_path (str): The path to the JSON file containing the API keys.
        key_name (str): The name of the specific API key to retrieve from the JSON file.

    Returns:
        str: The API key corresponding to the provided key name.

    Raises:
        KeyError: If the specified key name is not found in the JSON file.
        FileNotFoundError: If the JSON file is not found at the specified path.
        json.JSONDecodeError: If the JSON file is not properly formatted.
    """
    with open(json_file_path, "r") as file:
        data = json.load(file)
        return data[key_name]


def setup_logging(log_file="app.log"):
    """
    Set up logging configuration for the project.

    Args:
        log_file (str, optional): Path to the log file. Defaults to "app.log".
    """
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    log_file_path = os.path.join(logs_dir, log_file)

    # Remove any existing handlers attached to the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging with rotation
    handler = RotatingFileHandler(
        log_file_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,  # Keep 3 backups
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[handler, logging.StreamHandler()],
    )

    logging.info("Logging configuration complete.")
