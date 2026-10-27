# using ollama as a first base, we will get environment descriptions
# ! try other models
import multiprocessing as mp
import os
import sys
import base64
from typing import Optional

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import cv2
import numpy as np
from PIL import Image

from matplotlib import image
try:
    import ollama
except ImportError:
    ollama = None
import torch

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TextIteratorStreamer,
)

from src.tasks.config.utils import CONFIG, ENV_PROMPT, DEVICE

is_windows = sys.platform.startswith('win')

# vLLM path kept commented out; using the same Hugging Face workflow on Linux too.
# if not is_windows:
#     from vllm import LLM, SamplingParams

# Cache for loaded world models, keyed by model name to support multiple models
LOADED_MODEL = {}

## Query the model
def query_ollama_vlm(prompt=ENV_PROMPT, images=[], model=CONFIG["vlm"]["ollama"]["model_name"]):    # Convert images to base64 strings
    image_bytes = []
    for image in images:
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        image_bytes.append(image_base64)


    # Query the model
    response =ollama.generate(
        model=model,
        prompt=prompt,
        images=image_bytes,
        # temperature=CONFIG["vlm"]["temperature"],
    )
    return response


def load_world_model(model=CONFIG["vlm"]["world_model"]["model_name"]):
    """
    Loads the world model and caches it by its name to support switching between models.
    """
    # Load the world model
    if model in LOADED_MODEL:
        return LOADED_MODEL[model]

    # Use the same Hugging Face-based loading path on Linux as on Windows.
    world_model_processor = AutoProcessor.from_pretrained(model)
    world_model = AutoModelForImageTextToText.from_pretrained(
        model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(DEVICE)
    world_model.eval()
    world_model_tokenizer = world_model_processor.tokenizer
    LOADED_MODEL[model] = (world_model, world_model_processor, world_model_tokenizer)

    return world_model, world_model_processor, world_model_tokenizer

def unload_world_model(model_name: Optional[str] = None):
    """
    Removes a specific world model or all world models from the cache and clears VRAM.
    """
    global LOADED_MODEL
    if model_name:
        if model_name in LOADED_MODEL:
            del LOADED_MODEL[model_name]
    else:
        LOADED_MODEL.clear()

    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# * here we query the world model to describe the environment, we will be adding the segmentation of the static objects in the prompt here, will be completed next time
def query_world_model(
    prompt=ENV_PROMPT,
    images=None,
    model=CONFIG["vlm"]["world_model"]["model_name"],
    video_frames=None,
):
    # ! add here the static objects from segmentation
    # Load the world model
    world_model = load_world_model(model=model)

    content = []
    if video_frames is not None:
        video_frames = list(video_frames)
        print(f"DEBUG: query_world_model video frame count: {len(video_frames)}")
        if video_frames:
            rgb_video = np.stack(
                [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video_frames]
            )
            content.append({"type": "video", "video": rgb_video})
    else:
        images = images or []
        print(f"DEBUG: query_world_model images count: {len(images)}")
        if images:
            print(f"DEBUG: first image type: {type(images[0])}")
        pil_images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
        content.extend({"type": "image", "image": img} for img in pil_images)

    # Use the same Hugging Face generation flow on Linux as on Windows.
    content.append({"type": "text", "text": prompt})
    message = [
        {
            "role": "user",
            "content": content
        }
    ]

    # Query the model
    inputs = world_model[1].apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,  # add to config later
    ).to(DEVICE)

    # 1. Generate token output sequence
    with torch.inference_mode():
        outputs = world_model[0].generate(
            **inputs,
            max_new_tokens=1024,   # Adjust based on how long you want the generated output to be
            do_sample=True,       # Set to False if you want deterministic greedy search
            temperature=CONFIG["vlm"]["world_model"]["temperature"],
        )

    # 2. Slice off the input prompt tokens to keep only the newly generated text
    input_length = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][input_length:]

    # 3. Decode the token IDs back into string output
    response = world_model[1].decode(
        new_tokens,
        skip_special_tokens=True
    )

    # Explicitly delete GPU tensors to prevent memory leaks
    del inputs, outputs, new_tokens

    # Additional cleanup to ensure VRAM is released
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response
