import os
from pathlib import Path 
import logging

project_name = "churn-mlops"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/configs/__init__.py",
    f"src/{project_name}/configs/configurations.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "app.py",
    "requirements.txt",
    "setup.py",
    "research/research.ipynb",
    "templates/index.html",
    "test.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for the files: {filename}")
    if (not os.path.exists(filepath)):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Created file: {filepath}")

    else:
        logging.info(f"File already exists: {filepath}")