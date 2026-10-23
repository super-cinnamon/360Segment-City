
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class DramaLayout:
    """
    Adapter for the DRAMA dataset schema.
    All assumptions about the dataset layout are isolated here.
    """
    # ASSUMPTION: Root contains 'combined/' folder for frames and 'integrated_output_v2.json' for annotations.
    FRAMES_DIR = "combined"
    ANNOTATIONS_FILE = "integrated_output_v2.json"

    # ASSUMPTION: Frames are named frame_000001.png, etc.
    FRAME_PREFIX = "frame_"
    FRAME_EXT = ".png"

    @staticmethod
    def get_clip_path(root: str, clip_id: str) -> str:
        return os.path.join(root, DramaLayout.FRAMES_DIR, clip_id)

    @staticmethod
    def get_annotations_path(root: str) -> str:
        return os.path.join(root, DramaLayout.ANNOTATIONS_FILE)

class DramaLoader:
    def __init__(self, root: str):
        self.root = root
        self._validate_root()

    def _validate_root(self):
        if not self.root or self.root == "<SET_DRAMA_PATH_HERE>":
            raise ValueError("DRAMA dataset root is not set. Please set it in config_drama.yaml, "
                             "via --dataset-root, or DRAMA_ROOT env var.")

        root_path = Path(self.root)
        if not root_path.exists() or not root_path.is_dir():
            raise ValueError(f"DRAMA dataset root does not exist or is not a directory: {self.root}")

        # Check for the combined folder and annotations file
        if not (root_path / DramaLayout.FRAMES_DIR).exists():
            raise ValueError(f"Missing expected folder: {DramaLayout.FRAMES_DIR} in {self.root}")

        if not (root_path / DramaLayout.ANNOTATIONS_FILE).exists():
            raise ValueError(f"Missing expected annotations file: {DramaLayout.ANNOTATIONS_FILE} in {self.root}")

    def load_annotations(self) -> List[Dict[str, Any]]:
        path = DramaLayout.get_annotations_path(self.root)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def validate_dataset(self):
        """
        Reports class balance and metadata summary.
        """
        annos = self.load_annotations()
        total = len(annos)
        pos = sum(1 for a in annos if a.get("risk_label") == 1)
        neg = total - pos

        logger.info("--- DRAMA Dataset Validation ---")
        logger.info(f"Total clips: {total}")
        logger.info(f"Positive (Risk=1): {pos} ({pos/total:.1%})")
        logger.info(f"Negative (Risk=0): {neg} ({neg/total:.1%})")

        # Check for common fields
        if total > 0:
            sample = annos[0]
            fields = list(sample.keys())
            logger.info(f"Available fields: {fields}")

            required = ["clip_id", "risk_label", "bbox", "caption"]
            missing = [f for f in required if f not in fields]
            if missing:
                logger.error(f"Missing required fields in annotations: {missing}")
                return False, f"Missing fields: {missing}"

        return True, "Dataset validated successfully."
