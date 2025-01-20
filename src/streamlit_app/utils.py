import json


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
