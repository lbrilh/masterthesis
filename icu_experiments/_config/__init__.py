from pathlib import Path

from .load_config import load_config

config = {}

'''for p in Path(__file__).parent.glob("*.json"):
    config_ = load_config(p)
    if Path(config_["mlflow"]["tracking_uri"].replace("file:", "")).exists():
        config = config_'''
