import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import torch
from tqdm import tqdm

from src.tasks.config.utils import CONFIG, DEVICE


# Lazy-loaded globals to avoid allocating large models at import time
_segmentation_processor = None
_segmentation_model = None


def _ensure_segmentation_model():
    """Load the segmentation model and processor on first use."""
    global _segmentation_processor, _segmentation_model
    if _segmentation_processor is None or _segmentation_model is None:
        _segmentation_processor = OneFormerProcessor.from_pretrained(CONFIG["segmentation"]["model_name"])
        _segmentation_model = OneFormerForUniversalSegmentation.from_pretrained(CONFIG["segmentation"]["model_name"])  # load on CPU first
        # move to device only when needed
        _segmentation_model.to(DEVICE)


def predict_segmentation(image, task=CONFIG["segmentation"]["task"]):
    """Predict segmentation for a single image using a lazily-loaded model."""
    _ensure_segmentation_model()

    # Use globals
    processor = _segmentation_processor
    model = _segmentation_model

    target_size = [(image.shape[0], image.shape[1])]
    inputs = processor(images=image, task_inputs=[task], return_tensors="pt")
    # move tensors to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # pass through image_processor for postprocessing
    predicted_labels = None
    if task == "semantic":
        predicted_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.shape[:2]])[0]
    elif task == "instance":
        results = processor.post_process_instance_segmentation(outputs, target_sizes=[image.shape[:2]])[0]
        predicted_map = results["segmentation"]
        predicted_labels = results["segments_info"]
    elif task == "panoptic":
        results = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.shape[:2]])[0]
        predicted_map = results["segmentation"]
        predicted_labels = results["segments_info"]
    else:
        predicted_map = None

    predicted_map = predicted_map.cpu().numpy() if predicted_map is not None else None

    # cleanup
    del outputs, inputs

    return {
        "segmentation_map": predicted_map,
        "segmentation_labels": predicted_labels,
    }


def predict_segmentations(images, task=CONFIG["segmentation"]["task"]):
    """Predict segmentations for a list of images sequentially to avoid concurrent GPU spikes."""
    results = []
    for image in tqdm(images, desc="Segmenting images"):
        result = predict_segmentation(image, task=task)
        results.append(result)

    return results


def predict_cubic_segmentations(cubic_frames):
    """Process each cube side sequentially to avoid running multiple GPU tasks in parallel."""
    left_segmentations = predict_segmentations(cubic_frames["left"])
    right_segmentations = predict_segmentations(cubic_frames["right"])
    front_segmentations = predict_segmentations(cubic_frames["front"])
    back_segmentations = predict_segmentations(cubic_frames["back"])

    return {
        "left": left_segmentations,
        "right": right_segmentations,
        "front": front_segmentations,
        "back": back_segmentations,
    }
