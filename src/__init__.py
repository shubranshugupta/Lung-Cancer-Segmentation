import os
from os.path import dirname

ROOT_FOLDER = dirname(dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT_FOLDER, "logs")
DATA_DIR = os.path.join(ROOT_FOLDER, "data")
MODEL_DIR = os.path.join(ROOT_FOLDER, "models")

def create_dir() -> None:
    """
    A function to setup the Directory
    """
    # Checking for folder
    dir = [DATA_DIR, LOG_DIR, MODEL_DIR]
    for path in dir:
        os.makedirs(path, exist_ok=True)

create_dir()