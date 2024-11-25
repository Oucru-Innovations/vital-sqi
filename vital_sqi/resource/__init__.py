# resource/__init__.py

import os
import json
import logging

# Configure logging for the module
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Paths to the JSON files
SQI_DICT_PATH = os.path.join(os.path.dirname(__file__), "sqi_dict.json")
RULE_DICT_PATH = os.path.join(os.path.dirname(__file__), "rule_dict.json")


# Load the JSON files as dictionaries
def load_json(file_path, name):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error("Error loading %s: %s", name, e)
        return {}


# Load as dictionaries
sqi_dict = load_json(SQI_DICT_PATH, "sqi_dict.json")
rule_dict = load_json(RULE_DICT_PATH, "rule_dict.json")
