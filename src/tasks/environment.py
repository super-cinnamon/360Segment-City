# using ollama as a first base, we will get environment descriptions
# ! try other models
import multiprocessing as mp
import os
import sys
import base64
import tempfile
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
from openai import OpenAI

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TextIteratorStreamer,
)

from src.tasks.config.utils import CONFIG, ENV_PROMPT, DEVICE, resolve_vlm_profile

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


def load_world_model(model=None):
    """
    Loads the configured VLM profile and caches it by profile ID.
    """
    profile = resolve_vlm_profile(model)
    cache_key = profile["id"]
    if cache_key in LOADED_MODEL:
        return LOADED_MODEL[cache_key]

    repo_id = profile["repo_id"]
    if profile["backend"] == "llama.cpp":
        raise RuntimeError(
            f"VLM profile '{cache_key}' uses GGUF and must be served by the API's "
            "llama.cpp backend; it cannot be loaded in-process."
        )

    if profile["backend"] == "unsloth":
        try:
            from unsloth import FastVisionModel
        except ImportError as exc:
            raise RuntimeError(
                "This VLM profile requires Unsloth. Install the optional Unsloth "
                "runtime in the active environment."
            ) from exc

        world_model, world_model_processor = FastVisionModel.from_pretrained(
            model_name=repo_id,
            **profile.get("load_options", {}),
        )
        FastVisionModel.for_inference(world_model)
    else:
        world_model_processor = AutoProcessor.from_pretrained(repo_id)
        world_model = AutoModelForImageTextToText.from_pretrained(
            repo_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            **profile.get("load_options", {}),
        ).to(DEVICE)

    world_model.eval()
    world_model_tokenizer = getattr(world_model_processor, "tokenizer", world_model_processor)
    LOADED_MODEL[cache_key] = (world_model, world_model_processor, world_model_tokenizer)

    return world_model, world_model_processor, world_model_tokenizer

def unload_world_model(model_name: Optional[str] = None):
    """
    Removes a specific world model or all world models from the cache and clears VRAM.
    """
    global LOADED_MODEL
    if model_name:
        cache_key = model_name
        try:
            cache_key = resolve_vlm_profile(model_name)["id"]
        except ValueError:
            pass
        LOADED_MODEL.pop(cache_key, None)
    else:
        LOADED_MODEL.clear()

    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _query_gguf_world_model(prompt, images, video_frames, profile):
    content = [{"type": "text", "text": prompt}]
    if video_frames is not None:
        video_frames = list(video_frames)
        if video_frames:
            height, width = video_frames[0].shape[:2]
            if any(frame.shape[:2] != (height, width) for frame in video_frames):
                raise ValueError("All video frames must have the same dimensions.")

            descriptor, video_path = tempfile.mkstemp(suffix=".mp4")
            os.close(descriptor)
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                1.0,
                (width, height),
            )
            if not writer.isOpened():
                os.unlink(video_path)
                raise RuntimeError("OpenCV could not create a temporary MP4 for GGUF inference.")
            try:
                for frame in video_frames:
                    writer.write(frame)
            finally:
                writer.release()
            try:
                with open(video_path, "rb") as video_file:
                    video_base64 = base64.b64encode(video_file.read()).decode("ascii")
            finally:
                os.unlink(video_path)
            content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
            })
    else:
        for image in images or []:
            encoded, buffer = cv2.imencode(".jpg", image)
            if not encoded:
                raise RuntimeError("OpenCV could not encode an image for GGUF inference.")
            image_base64 = base64.b64encode(buffer).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            })

    client = OpenAI(
        base_url=CONFIG["risk_assessment"]["api_base"],
        api_key="vllm-local",
    )
    response = client.chat.completions.create(
        model=profile["repo_id"],
        messages=[{"role": "user", "content": content}],
        temperature=CONFIG["vlm"]["world_model"]["temperature"],
        max_tokens=1024,
    )
    generated_text = response.choices[0].message.content
    if not isinstance(generated_text, str):
        raise RuntimeError(
            f"GGUF profile '{profile['id']}' returned no text from the local API."
        )
    return generated_text


# * here we query the world model to describe the environment, we will be adding the segmentation of the static objects in the prompt here, will be completed next time
def query_world_model(
    prompt=ENV_PROMPT,
    images=None,
    model=None,
    video_frames=None,
):
    # ! add here the static objects from segmentation
    profile = resolve_vlm_profile(model)
    if profile["backend"] == "llama.cpp":
        return _query_gguf_world_model(prompt, images, video_frames, profile)

    # Load the in-process model
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
    processor_kwargs = {}
    if video_frames is not None and video_frames:
        video_processor = getattr(world_model[1], "video_processor", None)
        target_frame_count = getattr(video_processor, "num_frames", None)
        if target_frame_count is not None and len(video_frames) < target_frame_count:
            processor_kwargs["num_frames"] = len(video_frames)

    inputs = world_model[1].apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,  # add to config later
        processor_kwargs=processor_kwargs,
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
