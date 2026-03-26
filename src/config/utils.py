import json
import torch
import os


# load config
def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


CONFIG = load_config(os.path.join(os.path.dirname(__file__), "config.json"))

# check device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
