import json


# Load the API key from the JSON file
def load_api_key(json_file_path):
    with open(json_file_path, "r") as file:
        data = json.load(file)
        return data["api_key"]
