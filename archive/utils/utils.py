import logging
import os

import yaml


def setup_logging(level=logging.DEBUG):
    """Sets up logging with a defined format and specified level.

    Args:
        level: The logging level to use (default: logging.DEBUG)

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Return logger instance for convenience
    return logging.getLogger(__name__)


# Create a default logger instance
logger = logging.getLogger(__name__)


def load_config(config_file):
    """Loads configuration from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration data.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.

    Expected YAML structure:
        cryptocurrencies:
            - name: str
                symbol: str
        days: int
    """
    try:
        with open(config_file, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def validate_config(config):
    """Validates the required configuration values.

    Args:
        config (dict): Configuration data to validate.

    Raises:
        ValueError: If required configuration values are missing or invalid.
    """
    if not config.get("cryptocurrencies"):
        raise ValueError("Cryptocurrencies must be specified in the configuration.")
    if config.get("days", 0) <= 0:
        raise ValueError("Days must be a positive integer.")


def create_directories(directories):
    """Ensures specified directories exist.

    Args:
        directories (list): List of directory paths to create if they do not exist.

    Behavior:
        For each directory in the list, the function checks if the directory exists.
        If the directory does not exist, it creates the directory and logs the action.
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")
