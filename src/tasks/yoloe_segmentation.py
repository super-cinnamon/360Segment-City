"""
YOLOE Segmentation Module
=========================

Open-vocabulary segmentation using YOLOE-26-x-seg for hazard detection.
Replaces Grounded-SAM from the paper with a single model that does both
detection and segmentation via promptable classes.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import logging
import torch

from src.tasks.config.utils import CONFIG, DEVICE

logger = logging.getLogger(__name__)

# Global model cache
_yoloe_model = None


def _get_yoloe_model():
    """Lazy load YOLOE-26-x-seg model."""
    global _yoloe_model
    if _yoloe_model is None:
        from ultralytics import YOLO
        model_name = CONFIG["yoloe"].get("model_name", "yoloe-26x-seg.pt")
        logger.info(f"Loading YOLOE model: {model_name}")
        _yoloe_model = YOLO(model_name)
        _yoloe_model.to(DEVICE)
    return _yoloe_model


def unload_yoloe_model():
    """Unload YOLOE model and clear GPU memory."""
    global _yoloe_model
    if _yoloe_model is not None:
        del _yoloe_model
        _yoloe_model = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class YOLOESegmenter:
    """
    YOLOE-26-x-seg wrapper for open-vocabulary hazard segmentation.

    Uses promptable segmentation: set_classes() with hazard labels,
    then run inference to get masks, boxes, and confidences.
    """

    def __init__(self, confidence_threshold: float = None):
        self.confidence_threshold = confidence_threshold or CONFIG["yoloe"].get("confidence_threshold", 0.25)
        self.model = _get_yoloe_model()
        self._current_classes = None

    def set_classes(self, class_names: List[str]):
        """
        Set the classes for promptable segmentation.

        Args:
            class_names: List of class name strings (e.g., ["pothole", "wet road", "car"])
        """
        if class_names != self._current_classes:
            self.model.set_classes(class_names)
            self._current_classes = class_names.copy()
            logger.debug(f"YOLOE classes set to: {class_names}")

    def segment_hazards(
        self,
        image: np.ndarray,
        hazard_labels: List[str],
        confidence_threshold: float = None
    ) -> List[Dict]:
        """
        Segment specific hazard classes using YOLOE promptable segmentation.

        Args:
            image: RGB frame (H, W, 3) as numpy array
            hazard_labels: List of hazard class names from VLM
            confidence_threshold: Override default confidence threshold

        Returns:
            List of dicts with keys:
                - label: class name string
                - mask: binary mask (H, W) as bool array
                - confidence: detection confidence [0, 1]
                - bbox: [x1, y1, x2, y2] in pixel coordinates
                - class_id: integer class index
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to YOLOE segmenter")
            return []

        if not hazard_labels:
            logger.warning("No hazard labels provided to YOLOE segmenter")
            return []

        conf_thresh = confidence_threshold or self.confidence_threshold

        # Set classes for this inference
        self.set_classes(hazard_labels)

        # Run inference
        # YOLOE expects RGB, ultralytics handles BGR->RGB internally
        results = self.model(image, verbose=False, conf=conf_thresh)

        if not results or len(results) == 0:
            return []

        result = results[0]
        detections = []

        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data  # (N, H, W) tensor
            boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            confidences = result.boxes.conf.cpu().numpy()  # (N,)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

            # Get class names from model
            names = self.model.model.names

            for i in range(len(masks)):
                conf = float(confidences[i])
                if conf < conf_thresh:
                    continue

                class_id = class_ids[i]
                label = names.get(class_id, f"class_{class_id}")

                # Convert mask to numpy bool array
                mask_tensor = masks[i]
                if isinstance(mask_tensor, torch.Tensor):
                    mask = mask_tensor.cpu().numpy()
                else:
                    mask = mask_tensor

                # Resize mask to original image size if needed
                h, w = image.shape[:2]
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

                mask_bool = mask > 0.5

                # Get bbox
                bbox = boxes[i].astype(int).tolist()  # [x1, y1, x2, y2]

                detections.append({
                    "label": label,
                    "mask": mask_bool,
                    "confidence": conf,
                    "bbox": bbox,
                    "class_id": class_id
                })

        logger.debug(f"YOLOE detected {len(detections)} hazards: {[d['label'] for d in detections]}")
        return detections

    def segment_with_fixed_classes(
        self,
        image: np.ndarray,
        class_names: List[str],
        confidence_threshold: float = None
    ) -> List[Dict]:
        """
        Segment with a fixed set of classes (useful for batch processing).

        Args:
            image: RGB frame (H, W, 3)
            class_names: Fixed list of class names
            confidence_threshold: Override default

        Returns:
            List of detection dicts
        """
        self.set_classes(class_names)
        return self.segment_hazards(image, class_names, confidence_threshold)


def create_yoloe_segmenter(confidence_threshold: float = None) -> YOLOESegmenter:
    """Factory function to create YOLOE segmenter."""
    return YOLOESegmenter(confidence_threshold=confidence_threshold)