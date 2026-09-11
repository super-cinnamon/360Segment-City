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


def load_pipeline(
    model_name=CONFIG["depth_estimation"]["model_name"],
    device=DEVICE
    ):
    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        custom_pipeline="marigold_depth_estimation"
    )

    pipe = pipe.to(device)

    return pipe


# Lazy pipeline to avoid consuming GPU at import time
_depth_estimation_pipeline = None


def _get_depth_pipeline():
    global _depth_estimation_pipeline
    if _depth_estimation_pipeline is None:
        _depth_estimation_pipeline = load_pipeline()
    return _depth_estimation_pipeline


def predict_depth(image, pipe=None, hyperparameters=CONFIG["depth_estimation"]["hyperparameters"]):
    # Ablation: depth estimation disabled
    depth_pred = np.full(image.shape[:2], 0.5)
    depth_colored = Image.new('RGB', (image.shape[1], image.shape[0]), (0, 0, 0))
    return depth_colored, depth_pred


def predict_depths(images, pipe=None, hyperparameters=CONFIG["depth_estimation"]["hyperparameters"]):
    depth_output_images = []
    depth_output_predictions = []
    for input_image in tqdm(images, desc=f"Estimating depth (Ablation: Disabled)", leave=True):
        # Ablation: generate dummy values
        depth_output_image = Image.new('RGB', (input_image.shape[1], input_image.shape[0]), (0, 0, 0))
        depth_output_prediction = np.full(input_image.shape[:2], 0.5)
        depth_output_images.append(depth_output_image)
        depth_output_predictions.append(depth_output_prediction)

    return depth_output_images, depth_output_predictions


def predict_cubic_depths(cubic_frames, pipe=None, PARALLEL=False):
    # Ablation: return dummy depths for all sides
    return {
        side: predict_depths(frames)
        for side, frames in cubic_frames.items()
    }


def get_closest_depth_mask(depth_images, threshold=10):
    # Ablation: return mask of ones to avoid pruning segmentation
    if not depth_images:
        return None
    first_img = depth_images[0]
    return np.ones(first_img.shape[:2], dtype=np.uint8) * 255


def mode_depth(depth_map, segmentation_mask):
    # Ablation: return constant depth
    return 0.5
