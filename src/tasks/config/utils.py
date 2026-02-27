import json
import torch


# load config
def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


CONFIG = load_config("config.json")

# check device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
