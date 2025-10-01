import yaml
from pathlib import Path

def load_config(path="config.yaml"):
    with open(Path(path), "r") as f:
        config = yaml.safe_load(f)
    return config
