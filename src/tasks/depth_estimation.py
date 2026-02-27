import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from diffusers import DiffusionPipeline

from src.tasks.config.utils import CONFIG, DEVICE


def load_pipeline(
    model_name=CONFIG["depth_estimation"]["model_name"],
    pipeline_task=CONFIG["depth_estimation"]["pipeline_task"],
    device=DEVICE
    ):
    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        custom_pipeline="marigold_depth_estimation"
    )

    pipe = pipe.to(device)

    return pipe


def predict_depths(pipe, images, hyperparameters=CONFIG["depth_estimation"]["hyperparameters"]):
    depth_output_images = []
    with torch.no_grad():
        for input_image in tqdm(images, desc=f"Estimating depth", leave=True):
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

            # Save to the list
            depth_output_images.append(depth_colored)
    return depth_output_images


def predict_cubic_depths(pipe, cubic_frames):
    # parallel process all of the sides and recompile them into a list of dicts
    left_depths = predict_depths(pipe, cubic_frames["left"])
    right_depths = predict_depths(pipe, cubic_frames["right"])
    front_depths = predict_depths(pipe, cubic_frames["front"])
    back_depths = predict_depths(pipe, cubic_frames["back"])

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