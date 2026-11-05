import json
import os
import logging
from collections import Counter
from typing import Optional, Sequence

import torch

logger = logging.getLogger(__name__)


# load config
def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


CONFIG = load_config(os.path.join(os.path.dirname(__file__), "config.json"))

with open(os.path.join(os.path.dirname(__file__), "environment_prompt.md"), "r") as f:
    ENV_PROMPT = f.read()


def resolve_vlm_profile(profile_id: Optional[str] = None, config: Optional[dict] = None) -> dict:
    """Resolve a configured VLM profile and validate its runtime prerequisites."""
    config = CONFIG if config is None else config
    vlm_config = config.get("vlm", {})
    profiles = vlm_config.get("models", {})

    if not profiles:
        legacy_model = vlm_config.get("world_model", {}).get("model_name")
        if not legacy_model:
            raise ValueError("No VLM profiles or legacy world-model name are configured.")
        return {
            "id": legacy_model,
            "label": legacy_model,
            "repo_id": legacy_model,
            "backend": "transformers",
            "supports_video": None,
            "load_options": {},
        }

    selected_id = profile_id or vlm_config.get("selected_model")
    if not selected_id:
        raise ValueError("Set vlm.selected_model in config or pass --model.")
    if selected_id not in profiles:
        matching_id = next(
            (key for key, value in profiles.items() if value.get("repo_id") == selected_id),
            None,
        )
        if matching_id is None:
            raise ValueError(
                f"Unknown VLM profile '{selected_id}'. Available profiles: "
                f"{', '.join(profiles)}"
            )
        selected_id = matching_id

    profile = dict(profiles[selected_id])
    repo_id = profile.get("repo_id", "").strip()
    if not repo_id:
        raise ValueError(
            f"VLM profile '{selected_id}' has no repo_id; set it in "
            "src/tasks/config/config.json before running."
        )
    if profile.get("supports_video") is False:
        raise ValueError(
            f"VLM profile '{selected_id}' does not support the pipeline's required video input."
        )
    if profile.get("supports_video") is not True:
        raise ValueError(
            f"Video support for VLM profile '{selected_id}' is not verified. "
            "Set supports_video to true only after confirming the selected checkpoint."
        )
    if profile.get("backend") not in {"unsloth", "transformers", "llama.cpp"}:
        raise ValueError(
            f"Unsupported backend '{profile.get('backend')}' for VLM profile '{selected_id}'."
        )
    if profile.get("artifact_format") == "gguf" and profile.get("backend") != "llama.cpp":
        raise ValueError(
            f"VLM profile '{selected_id}' uses GGUF but is not configured for the "
            "'llama.cpp' backend."
        )
    if profile.get("backend") == "llama.cpp" and profile.get("artifact_format") != "gguf":
        raise ValueError(
            f"VLM profile '{selected_id}' uses the 'llama.cpp' backend but is not "
            "configured as a GGUF artifact."
        )

    profile["id"] = selected_id
    profile["repo_id"] = repo_id
    profile.setdefault("label", selected_id)
    profile.setdefault("load_options", {})
    return profile


def apply_vlm_profile(profile_id: Optional[str] = None, config: Optional[dict] = None) -> dict:
    """Apply one validated profile to all pipeline model-name settings in memory."""
    config = CONFIG if config is None else config
    profile = resolve_vlm_profile(profile_id, config)
    repo_id = profile["repo_id"]

    for section in ("vlm", "risk_assessment", "risk_map"):
        config.setdefault(section, {})["model_name"] = repo_id
    config["vlm"].setdefault("world_model", {})["model_name"] = repo_id
    return profile


def format_static_objects_summary(static_objects: Optional[Sequence[dict]]) -> str:
    if not static_objects:
        return "[]"
    return str([
        {"class_name": item.get("class_name"), "count": item.get("count", 1)}
        for item in static_objects
    ])


def render_environment_prompt(prompt: str, static_objects: Optional[Sequence[dict]] = None) -> str:
    """Render the environment prompt using .replace() to avoid conflicts with JSON curly braces."""
    if not prompt:
        return ""

    static_objects_summary = format_static_objects_summary(static_objects)
    return prompt.replace("{STATIC_OBJECTS}", static_objects_summary)


# check device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
