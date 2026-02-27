import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_video():
    pass


def split_frames(video_path, threshold=2.0, max_to_extract=500):
    cap = cv2.VideoCapture(video_path)
    unique_frames = []
    last_frame_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(unique_frames) >= max_to_extract:
            break

        # Convert to grayscale for comparison logic
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if last_frame_gray is None:
            is_different = True
        else:
            mse = np.mean((gray_frame.astype("float") - last_frame_gray.astype("float")) ** 2)
            is_different = mse > threshold

        if is_different:
            # Store the COLOR frame, but convert BGR to RGB for Matplotlib
            unique_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            last_frame_gray = gray_frame

    cap.release()
    print(f"Extracted {len(unique_frames)} unique frames into memory.")
    return unique_frames


def load_frames():
    pass


def save_frames():
    pass


def load_cubic(cubic_path_root, prefix="frame_"):
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
            cubic_frames["back"].append(cv2.flip(cv2.imread(f"{cubic_path_root}/{folder_name}/back.png"), 1))
            cubic_frames["front"].append(cv2.flip(cv2.imread(f"{cubic_path_root}/{folder_name}/front.png"), 1))

            # left and right are flipped due to mirroring effect
            cubic_frames["right"].append(cv2.flip(cv2.imread(f"{cubic_path_root}/{folder_name}/left.png"), 1))
            cubic_frames["left"].append(cv2.flip(cv2.imread(f"{cubic_path_root}/{folder_name}/right.png"), 1))
        else:
            print(f"Skipping {folder_name} (does not exist)")


def display_cubic(cubic_frames):
    pass
