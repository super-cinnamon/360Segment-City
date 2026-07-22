# using ollama as a first base, we will get environment descriptions
import sys
import base64
import cv2
from PIL import Image

from matplotlib import image
import ollama
import torch

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TextIteratorStreamer,
)

from src.tasks.config.utils import CONFIG, ENV_PROMPT, DEVICE

is_windows = sys.platform.startswith('win')

if not is_windows:
    from vllm import LLM, SamplingParams

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
    # Load the world model
    if "world_model" in LOADED_MODEL:
        return LOADED_MODEL[model]
    # Load model onto local hardware via vLLM engine if linux, on huggingface if windows
    if is_windows:
        world_model_processor = AutoProcessor.from_pretrained(model)
        world_model = AutoModelForImageTextToText.from_pretrained(
            model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(DEVICE)
        world_model.eval()
        LOADED_MODEL["world_model"] = (world_model, world_model_processor)
        return world_model, world_model_processor

    else:
        world_model = LLM(
            model=model,
            tensor_parallel_size=1, # Set to >1 if using multiple GPUs
            max_model_len=4096,
            trust_remote_code=True
        )
        LOADED_MODEL["world_model"] = world_model
        return world_model


def query_world_model(prompt=ENV_PROMPT, images=[], model=CONFIG["vlm"]["world_model"]["model_name"]):
    # Load the world model
    world_model = load_world_model(model=model)

    # Convert images to PIL format
    pil_images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]

    if is_windows:
        content = [{"type": "image", "image": img} for img in pil_images]
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
                max_new_tokens=512,   # Adjust based on how long you want the generated output to be
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
    
    else:
        sampling_params = SamplingParams(temperature=CONFIG["vlm"]["world_model"]["temperature"], max_tokens=512)

        # Query the model
        inputs = {
            "prompt": "<image>\n" + prompt,
            "multi_modal_data": {"image": pil_images[0]},
        }

        outputs = world_model.generate([inputs], sampling_params)

        response = [output.outputs[0].text for output in outputs]

    return response
