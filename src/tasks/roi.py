import json
import re
from collections import Counter
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


ROI_KEYWORDS: list[tuple[str, float]] = [
    ("crossroad", 1.25),
    ("crossroads", 1.25),
    ("roundabout", 1.2),
    ("traffic light", 1.0),
    ("traffic lights", 1.0),
    ("pedestrian crossing", 1.1),
    ("crosswalk", 1.1),
    ("pedestrian", 0.8),
    ("pedestrians", 0.8),
    ("heavy traffic", 1.2),
    ("high density traffic", 1.2),
    ("traffic", 0.5),
    ("crowded", 0.7),
    ("congestion", 0.7),
]


def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_environment_risk_score(description: Optional[str]) -> Optional[float]:
    if description is None:
        return None

    if isinstance(description, (int, float)) and not isinstance(description, bool):
        return float(description)

    if isinstance(description, dict):
        for key in ("environment_risk_score", "roi_score", "risk_score", "score"):
            value = description.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return None

    text = str(description).strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None

    if parsed is not None:
        parsed_score = _extract_environment_risk_score(parsed)
        if parsed_score is not None:
            return parsed_score

    for key in ("environment_risk_score", "roi_score", "risk_score", "score"):
        match = re.search(rf'"{re.escape(key)}"\s*:\s*([-+]?\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))

    return None


def compute_roi_score(description: Optional[str]) -> Tuple[float, list[str]]:
    """Score the likelihood that the environment description represents a high-risk ROI."""
    cleaned = _normalize_text(description)
    if not cleaned:
        return 0.0, []

    matched_keywords: list[str] = []
    keyword_score = 0.0
    for keyword, weight in ROI_KEYWORDS:
        if keyword in cleaned:
            matched_keywords.append(keyword)
            keyword_score += weight

    normalized_keyword_score = min(1.0, keyword_score / 4.0)

    parsed_score = _extract_environment_risk_score(description)
    if parsed_score is not None:
        if parsed_score > 1.0:
            parsed_score = min(1.0, parsed_score / 5.0)
        parsed_score = min(1.0, max(0.0, parsed_score))
        blended_score = min(1.0, 0.7 * parsed_score + 0.3 * normalized_keyword_score)
        combined_score = round(max(parsed_score, blended_score), 4)
        return combined_score, matched_keywords

    return round(normalized_keyword_score, 4), matched_keywords


def should_process_window(
    description: Optional[str],
    roi_enabled: bool = True,
    roi_threshold: float = 0.75,
) -> bool:
    if not roi_enabled:
        return True
    score, _ = compute_roi_score(description)
    return score >= roi_threshold


def resize_frame(frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if frame is None:
        return frame
    if scale is None:
        return frame
    if scale <= 0.0:
        return frame
    if scale >= 1.0:
        return frame

    height, width = frame.shape[:2]
    if width == 0 or height == 0:
        return frame
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size)


def resize_frames(frames: Sequence[np.ndarray], scale: float = 1.0) -> list[np.ndarray]:
    if frames is None:
        return []
    return [resize_frame(frame, scale=scale) for frame in frames]


def build_static_objects_summary(environment_items: Iterable[Sequence[dict]]) -> list[dict]:
    """Summarise static roadway objects detected in a window for prompt augmentation."""
    counter = Counter()
    for frame_segments in environment_items:
        for segment in frame_segments:
            class_name = segment.get("class_name")
            if class_name:
                counter[class_name] += 1

    return [
        {"class_name": class_name, "count": count}
        for class_name, count in sorted(counter.items())
    ]


def format_static_objects_summary(static_objects: Optional[Sequence[dict]]) -> str:
    if not static_objects:
        return "[]"
    return str([
        {"class_name": item.get("class_name"), "count": item.get("count", 1)}
        for item in static_objects
    ])
