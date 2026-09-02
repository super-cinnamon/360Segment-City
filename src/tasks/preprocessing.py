import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm
import py360convert

from src.tasks.config.utils import CONFIG


def load_video():
    pass


def split_frames(video_path, threshold=None, max_to_extract=2000, prune_similar_frames: Optional[bool] = None):
    if threshold is None:
        threshold = CONFIG["processing"].get("frame_similarity_threshold", 2.0)
    if prune_similar_frames is None:
        prune_similar_frames = CONFIG["processing"].get("prune_similar_frames", False)

    cap = cv2.VideoCapture(video_path)
    unique_frames = []
    last_frame_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(unique_frames) >= max_to_extract:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prune_similar_frames:
            if last_frame_gray is None:
                is_different = True
            else:
                mse = np.mean((gray_frame.astype("float") - last_frame_gray.astype("float")) ** 2)
                is_different = mse > threshold
        else:
            is_different = True

        if is_different:
            unique_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            last_frame_gray = gray_frame

    cap.release()
    print(f"Extracted {len(unique_frames)} unique frames into memory.")
    return unique_frames


def split_frames_window(video_path, start_frame=0, threshold=None, max_to_extract=2000, prune_similar_frames: Optional[bool] = None):
    """
    Reads a video starting at `start_frame` (0-based raw frame index) and
    extracts up to `max_to_extract` unique frames using the same uniqueness
    thresholding logic as `split_frames` unless frame pruning is disabled.

    Returns a tuple `(unique_frames, last_raw_index)` where `last_raw_index` is
    the last raw frame index read from the video (0-based). If EOF is reached
    immediately, returns ([], None).
    """
    if threshold is None:
        threshold = CONFIG["processing"].get("frame_similarity_threshold", 2.0)
    if prune_similar_frames is None:
        prune_similar_frames = CONFIG["processing"].get("prune_similar_frames", False)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if start_frame >= total_frames:
        cap.release()
        return [], None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    unique_frames = []
    last_frame_gray = None
    last_raw_index = start_frame - 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        last_raw_index += 1

        if len(unique_frames) >= max_to_extract:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prune_similar_frames:
            if last_frame_gray is None:
                is_different = True
            else:
                mse = np.mean((gray_frame.astype("float") - last_frame_gray.astype("float")) ** 2)
                is_different = mse > threshold
        else:
            is_different = True

        if is_different:
            unique_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            last_frame_gray = gray_frame

    cap.release()
    if unique_frames:
        print(f"Window start={start_frame}: extracted {len(unique_frames)} unique frames (last_raw_index={last_raw_index})")
        return unique_frames, last_raw_index
    else:
        return [], None


def generate_cubic(frames, face_w=512):
    """Converts a list of 360/equirectangular frame images (NumPy arrays)

    into a dictionary containing 'left', 'right', 'front', and 'back' face lists.
    """
    cubic_frames = {
        "left": [],
        "right": [],
        "front": [],
        "back": [],
    }

    for frame in frames:
        # e2c returns a dictionary with keys: 'F', 'R', 'B', 'L', 'U', 'D'
        cube_dict = py360convert.e2c(
            frame, cube_format="dict", face_w=face_w, mode="bilinear"
        )

        cubic_frames["front"].append(cube_dict["F"])
        cubic_frames["right"].append(cube_dict["R"])
        cubic_frames["back"].append(cube_dict["B"])
        cubic_frames["left"].append(cube_dict["L"])
    # erase original equirectangular frames from memory
    del frames

    return cubic_frames


def load_frames():
    pass


def save_frames():
    pass


def load_cubic(cubic_path_root, prefix="frame_", resizing_factor=0.5):
    cubic_frames = {
        "left": [],
        "right": [],
        "front": [],
        "back": [],
    }

    range_folders = len([f for f in os.listdir(cubic_path_root) if os.path.isdir(os.path.join(cubic_path_root, f))])

    # make it use tqdm
    for i in tqdm(range(range_folders), desc="Loading cubic frames"):
        # :06d means: integer, padded with zeros to 6 digits
        folder_name = f"{prefix}{i:06d}"

        folder_path = Path(cubic_path_root + "/" + folder_name)

        if folder_path.exists():
            # print(f"Found {folder_name}")
            cubic_frames["back"].append(cv2.resize(cv2.flip(cv2.imread(f"{cubic_path_root}/{folder_name}/back.png"), 1), (0, 0), fx=resizing_factor, fy=resizing_factor))
            cubic_frames["front"].append(cv2.resize(cv2.flip(cv2.imread(f"{cubic_path_root}/{folder_name}/front.png"), 1), (0, 0), fx=resizing_factor, fy=resizing_factor))

            # left and right are flipped due to mirroring effect
            cubic_frames["right"].append(cv2.resize(cv2.flip(cv2.imread(f"{cubic_path_root}/{folder_name}/left.png"), 1), (0, 0), fx=resizing_factor, fy=resizing_factor))
            cubic_frames["left"].append(cv2.resize(cv2.flip(cv2.imread(f"{cubic_path_root}/{folder_name}/right.png"), 1), (0, 0), fx=resizing_factor, fy=resizing_factor))
        else:
            print(f"Skipping {folder_name} (does not exist)")
    
    return cubic_frames


def display_cubic(cubic_frames):
    pass
