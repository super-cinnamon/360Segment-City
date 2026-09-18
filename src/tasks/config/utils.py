import json
import os
from collections import Counter
from typing import Optional, Sequence

import torch


# load config
def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


CONFIG = load_config(os.path.join(os.path.dirname(__file__), "config.json"))

with open(os.path.join(os.path.dirname(__file__), "environment_prompt.md"), "r") as f:
    ENV_PROMPT = f.read()


def format_static_objects_summary(static_objects: Optional[Sequence[dict]]) -> str:
    if not static_objects:
        return "[]"
    return str([
        {"class_name": item.get("class_name"), "count": item.get("count", 1)}
        for item in static_objects
    ])


def render_environment_prompt(prompt: str, static_objects: Optional[Sequence[dict]] = None) -> str:
    """Render the environment prompt without interpreting JSON-style braces in the template."""
    if not prompt:
        return ""

    static_objects_summary = format_static_objects_summary(static_objects)
    return prompt.replace("{STATIC_OBJECTS}", static_objects_summary)


# check device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
