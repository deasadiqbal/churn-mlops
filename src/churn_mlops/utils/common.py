import os
import logging
import yaml
from box import ConfigBox
from pathlib import Path
from churn_mlops import logger
from box.exceptions import BoxValueError

def read_yaml(file_path: Path) -> ConfigBox:
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
            return ConfigBox(config)
            logger.info(f"File read successfully")
    except BoxValueError:
        raise("Please provide a valid yaml file")
    except Exception as e:
        raise e