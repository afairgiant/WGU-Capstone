import logging
import os

import yaml


def setup_logging(level=logging.DEBUG):
    """Sets up logging with a defined format and specified level.

    Args:
        level: The logging level to use (default: logging.DEBUG)
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
    """Loads configuration from a YAML file."""
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
    """Validates the required configuration values."""
    if not config.get("cryptocurrencies"):
        raise ValueError("Cryptocurrencies must be specified in the configuration.")
    if config.get("days", 0) <= 0:
        raise ValueError("Days must be a positive integer.")


def create_directories(directories):
    """Ensures specified directories exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")
