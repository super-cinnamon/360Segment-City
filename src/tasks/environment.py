# using ollama as a first base, we will get environment descriptions
import base64
import cv2

import ollama

from src.tasks.config.utils import CONFIG, ENV_PROMPT


## Query the model
def query_vlm(prompt=ENV_PROMPT, images=[], model=CONFIG["vlm"]["model_name"]):
    # Convert images to base64 strings
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