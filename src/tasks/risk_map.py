"""
risk_map.py
===========
Implements the Risk Map Construction module based on the VLM-Based Advanced Rider Assistance
System (ARAS) for motorcycles.

This module computes a dense spatial risk map by aggregating several risk factors:
1. Contextual VLM Risk (c_vlm): Semantic reasoning from the world model.
2. Area-Based Risk (c_area): Sigmoid-normalized hazard area ratio.
3. Confidence Score (c_conf): Grounding model confidence.
4. Depth-Based Risk (c_depth): Normalized physical depth deviation.

The final risk map is aggregated by taking the maximum severity across overlapping hazards.
"""

import numpy as np
import cv2
import logging
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from src.tasks.config.utils import CONFIG
from src.tasks.depth_estimation import mode_depth
from src.tasks.environment import query_world_model

logger = logging.getLogger(__name__)

@dataclass
class Hazard:
    """Represents a detected hazard and its associated risk components."""
    id: str
    mask: np.ndarray
    c_vlm: float
    c_conf: float
    depth: float
    description: str

class RiskMapEngine:
    """
    Engine to construct the dense spatial risk map \mathcal{C}_{risk}.
    """
    def __init__(self):
        self.cfg = CONFIG.get("risk_map", {})
        self.weights = self.cfg.get("weights", {"vlm": 0.4, "area": 0.2, "conf": 0.2, "depth": 0.2})
        self.hyperparams = self.cfg.get("hyperparameters", {"kappa_a": 10.0, "eta_a": 0.05, "d_ref": 0.5})
        self.model_name = self.cfg.get("model_name", "nvidia/Cosmos3-Edge")
        self.prompt = self.cfg.get("prompt", "")

    def _compute_area_risk(self, mask: np.ndarray, h: int, w: int) -> float:
        """
        Equation (3): c_area = sigma(kappa_a * (area_ratio - eta_a))
        """
        area_ratio = np.sum(mask) / (h * w)
        kappa_a = self.hyperparams.get("kappa_a", 10.0)
        eta_a = self.hyperparams.get("eta_a", 0.05)

        # Sigmoid function
        z = kappa_a * (area_ratio - eta_a)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_depth_risk(self, depth_val: float) -> float:
        """
        Equation (4): c_depth = min(1, d_i / d_ref)
        """
        d_ref = self.hyperparams.get("d_ref", 0.5)
        return min(1.0, depth_val / d_ref)

    def generate_risk_map(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        hazards_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[Hazard]]:
        """
        Constructs the dense spatial risk map.

        Args:
            image: The RGB frame (H, W, 3).
            depth_map: The depth map (H, W).
            hazards_data: A list of hazards identified by the VLM/Grounding system.
                          Expected schema: [{"id": str, "mask": bool_array, "score": float, "description": str}]

        Returns:
            risk_map: Dense map of size (H, W).
            hazards: List of Hazard dataclasses with computed components.
        """
        h, w = image.shape[:2]
        risk_map = np.zeros((h, w), dtype=np.float32)
        computed_hazards = []

        for h_data in hazards_data:
            mask = h_data["mask"]
            if mask is None or not np.any(mask):
                continue

            # 1. Contextual VLM Risk (c_vlm)
            # In a real scenario, this comes from the VLM's structured output for this specific hazard.
            # Here we use the 'score' provided in hazards_data.
            c_vlm = float(h_data.get("score", 0.0))

            # 2. Area-Based Risk (c_area)
            c_area = self._compute_area_risk(mask, h, w)

            # 3. Confidence Score (c_conf)
            # This is the detection confidence from the grounding model.
            c_conf = float(h_data.get("conf", 0.0))

            # 4. Depth-Based Risk (c_depth)
            # Calculate mode depth for the hazard mask.
            d_i = mode_depth(depth_map, mask)
            c_depth = self._compute_depth_risk(d_i)

            # Total Scalar Cost C_i = sum(alpha * c)
            total_cost = (
                self.weights.get("vlm", 0.4) * c_vlm +
                self.weights.get("area", 0.2) * c_area +
                self.weights.get("conf", 0.2) * c_conf +
                self.weights.get("depth", 0.2) * c_depth
            )

            # Update risk map: C_risk(m, n) = max(C_risk, C_i)
            risk_map = np.maximum(risk_map, mask * total_cost)

            computed_hazards.append(Hazard(
                id=h_data.get("id", "unknown"),
                mask=mask,
                c_vlm=c_vlm,
                c_conf=c_conf,
                depth=d_i,
                description=h_data.get("description", "")
            ))

        return risk_map, computed_hazards

    def identify_hazards_vlm(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Wraps the world model to identify hazards and their contextual risk.
        This function converts the VLM's text response into a list of hazard dicts.
        """
        # The prompt is defined in config.
        # We use the existing query_world_model infrastructure.
        response_text = query_world_model(
            prompt=self.prompt,
            images=[image],
            model=self.model_name
        )

        # This is a placeholder for the parsing logic.
        # In a production setting, the VLM would be prompted to return JSON.
        # For now, we'll assume a simple JSON parsing or dummy data if parsing fails.
        try:
            # Attempt to extract JSON from response
            import re
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0)).get("hazards", [])
        except Exception as e:
            logger.warning("Failed to parse VLM hazard response: %s", e)

        return []
