import gc

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import torch
from tqdm import tqdm

from src.tasks.config.utils import CONFIG, DEVICE


# TODO maybe move all to init or utils for avoiding reloads and circular imports
def load_model():
    processor = OneFormerProcessor.from_pretrained(CONFIG["segmentation"]["model_name"])
    model = OneFormerForUniversalSegmentation.from_pretrained(CONFIG["segmentation"]["model_name"]).to(DEVICE)

    return processor, model


segmentation_processor, segmentation_model = load_model()


def predict_segmentation(image, processor, model, task=CONFIG["segmentation"]["task"]):
    target_size = [(image.shape[0], image.shape[1])]
    inputs = processor(images=image, task_inputs=[task], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # pass through image_processor for postprocessing
    if task == "semantic":
        predicted_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.shape[:2]])[0]
    if task == "instance":
        predicted_map = processor.post_process_instance_segmentation(outputs, target_sizes=[image.shape[:2]])[0]["segmentation"]
    if task == "panoptic":
        predicted_map = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.shape[:2]])[0]["segmentation"]

    predicted_map = predicted_map.cpu().numpy()

    del outputs, inputs

    # 2. Force Python's Garbage Collector to run
    gc.collect()

    # 3. Clear the actual VRAM cache in the GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_segmentations(images, processor, model, task=CONFIG["segmentation"]["task"]):
    results = []
    for image in tqdm(images, desc=f"Segmenting images"):
        result = predict_segmentation(image, processor, model, task)
        results.append(result)
    return results
