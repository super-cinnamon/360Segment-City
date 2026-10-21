import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from diffusers import DiffusionPipeline
from scipy import stats

# for parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.tasks.config.utils import CONFIG, DEVICE


def get_depth_mode():
    mode = CONFIG["depth_estimation"].get("mode", "marigold")
    if not isinstance(mode, str) or mode.lower() not in {"marigold", "yolo"}:
        raise ValueError(
            "depth_estimation.mode must be either 'marigold' or 'yolo'"
        )
    return mode.lower()


def get_depth_model_config(mode=None):
    selected_mode = get_depth_mode() if mode is None else mode
    try:
        return CONFIG["depth_estimation"]["models"][selected_mode]
    except KeyError as error:
        raise ValueError(
            f"Missing configuration for depth estimation mode '{selected_mode}'"
        ) from error


def load_pipeline(model_config=None, device=DEVICE):
    if model_config is None:
        model_config = get_depth_model_config("marigold")

    pipe = DiffusionPipeline.from_pretrained(
        model_config["model_name"],
        custom_pipeline=model_config["pipeline_name"],
    )

    pipe = pipe.to(device)

    return pipe


# Lazy pipeline to avoid consuming GPU at import time
_depth_estimation_pipeline = None
_yolo_depth_model = None

def unload_model():
    global _depth_estimation_pipeline, _yolo_depth_model
    _depth_estimation_pipeline = None
    _yolo_depth_model = None
    import torch
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def _get_depth_pipeline():
    global _depth_estimation_pipeline
    if _depth_estimation_pipeline is None:
        _depth_estimation_pipeline = load_pipeline(
            get_depth_model_config("marigold")
        )
    return _depth_estimation_pipeline


def _get_yolo_depth_model():
    global _yolo_depth_model
    if _yolo_depth_model is None:
        from ultralytics import YOLO

        model_config = get_depth_model_config("yolo")
        _yolo_depth_model = YOLO(model_config["model_name"])
    return _yolo_depth_model


def _get_configured_depth_model():
    if get_depth_mode() == "yolo":
        return _get_yolo_depth_model()
    return _get_depth_pipeline()


def _prepare_yolo_depth(depth, image_shape):
    if hasattr(depth, "data"):
        depth = depth.data
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    depth = np.asarray(depth)
    while depth.ndim > 2 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D YOLO depth map, got shape {depth.shape}")

    valid = np.isfinite(depth)
    if not np.any(valid):
        raise ValueError("YOLO returned a depth map with no finite values")

    depth = depth.astype(np.float32, copy=True)
    depth[~valid] = float(depth[valid].max())

    height, width = image_shape[:2]
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)

    min_depth = float(depth.min())
    max_depth = float(depth.max())
    normalized_depth = np.ones((height, width), dtype=np.float32)
    if max_depth > min_depth:
        normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    else:
        normalized_depth.fill(0.0)
    return normalized_depth


def predict_depth(image, pipe=None, hyperparameters=None):
    # make image into PIL Image
    input_image = Image.fromarray(image)

    if get_depth_mode() == "yolo":
        if pipe is None:
            pipe = _get_configured_depth_model()

        with torch.no_grad():
            results = pipe(input_image, device=DEVICE, verbose=False)

        if not results or getattr(results[0], "depth", None) is None:
            raise ValueError("YOLO depth model did not return a depth map")

        depth_pred = _prepare_yolo_depth(results[0].depth, image.shape)
        from diffusers.pipelines.marigold.marigold_image_processing import (
            MarigoldImageProcessor,
        )

        depth_colored = MarigoldImageProcessor.visualize_depth(
            depth_pred, color_map="Spectral"
        )[0]
        return depth_colored, depth_pred

    if pipe is None:
        pipe = _get_depth_pipeline()

    if hyperparameters is None:
        hyperparameters = get_depth_model_config("marigold")["hyperparameters"]

    with torch.no_grad():
        # Predict depth
        pipeline_output = pipe(
            input_image,
            denoising_steps=hyperparameters["denoising_steps"],     # optional
            ensemble_size=hyperparameters["ensemble_size"],       # optional
            processing_res=hyperparameters["processing_res"],     # optional
            match_input_res=hyperparameters["match_input_res"],   # optional
            batch_size=0,           # optional
            color_map="Spectral",   # optional
            show_progress_bar=True, # optional
            # seed=seed,              # optional
        )

        depth_pred: np.ndarray = pipeline_output.depth_np
        depth_colored: Image.Image = pipeline_output.depth_colored

    return depth_colored, depth_pred


def predict_depths(images, pipe=None, hyperparameters=None):
    depth_output_images = []
    depth_output_predictions = []
    with torch.no_grad():
        if pipe is None:
            pipe = _get_configured_depth_model()

        for input_image in tqdm(images, desc=f"Estimating depth", leave=True):
            # Predict depth
            depth_output_image, depth_prediction = predict_depth(input_image, pipe, hyperparameters)

            # Save to the list
            depth_output_images.append(depth_output_image)
            depth_output_predictions.append(depth_prediction)

        return depth_output_images, depth_output_predictions


def predict_cubic_depths(cubic_frames, pipe=None, PARALLEL=False):
    if pipe is None:
        pipe = _get_configured_depth_model()

    left_depths = predict_depths(cubic_frames["left"], pipe)
    right_depths = predict_depths(cubic_frames["right"], pipe)
    front_depths = predict_depths(cubic_frames["front"], pipe)
    back_depths = predict_depths(cubic_frames["back"], pipe)

    return {
        "left": left_depths,
        "right": right_depths,
        "front": front_depths,
        "back": back_depths,
    }


def get_closest_depth_mask(depth_images, threshold=10):
    combined_mask = None

    for heatmap in depth_images:
        # 1. Ensure Grayscale
        if len(heatmap.shape) == 3:
            # Using the Green channel as per your original logic [:, :, 1]
            heatmap_gray = heatmap[:, :, 1]
        else:
            heatmap_gray = heatmap

        # 2. Threshold each heatmap
        _, current_mask = cv2.threshold(heatmap_gray, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY_INV)

        # 3. Combine masks using Bitwise AND (Intersection)
        if combined_mask is None:
            combined_mask = current_mask
        else:
            combined_mask = cv2.bitwise_and(combined_mask, current_mask)

    # 4. Refine the final consensus mask
    if combined_mask is not None:
        kernel = np.ones((20, 20), np.uint8)
        # Close holes to make solid chunks
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        # Dilate to expand the mask slightly
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    
    return combined_mask


def mode_depth(depth_map, segmentation_mask):
    # Ensure your mask is a boolean array (True for object pixels, False for background)
    binary_mask = segmentation_mask > 0

    # Filter depth map using the mask (returns a 1D array of depth values)
    object_depths = depth_map[binary_mask]

    # Round to discrete levels so identical float values can group up for a mode calculation
    binned_depths = np.round(object_depths, decimals=2)

    # Get the mode depth of the object
    mode_depth = stats.mode(binned_depths, keepdims=False).mode


    return mode_depth
