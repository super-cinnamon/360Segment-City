"""
risk_map.py
===========
Implements the Risk Map Construction module based on the VLM-Based Advanced Rider Assistance
System (ARAS) for motorcycles (Algorithm 1, Lines 1-6).

This module computes a dense spatial risk map by aggregating several risk factors:
1. Contextual VLM Risk (c_vlm): Semantic reasoning from VLM with motorcycle state.
2. Area-Based Risk (c_area): Sigmoid-normalized hazard area ratio.
3. Confidence Score (c_conf): Grounding model detection confidence.
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
    c_area: float = 0.0
    c_depth: float = 0.0
    total_cost: float = 0.0


class RiskMapEngine:
    """
    Engine to construct the dense spatial risk map C_risk.

    Implements Algorithm 1 from the paper:
    - Line 2-6: For each hazard, compute all four risk components and total cost
    - Line 8: Pixel-wise max fusion across overlapping hazards
    """

    def __init__(self, model_name: Optional[str] = None):
        self.cfg = CONFIG.get("risk_map", {})
        self.weights = self.cfg.get("weights", {"vlm": 0.4, "area": 0.2, "conf": 0.2, "depth": 0.2})
        self.hyperparams = self.cfg.get("hyperparameters", {"kappa_a": 10.0, "eta_a": 0.05, "d_ref": 0.5})
        self.model_name = model_name or self.cfg.get("model_name", "nvidia/Cosmos3-Nano")
        self.prompt = self.cfg.get("prompt", "")

    def _compute_area_risk(self, mask: np.ndarray, h: int, w: int) -> float:
        """
        Equation (3): c_area = sigma(kappa_a * (area_ratio - eta_a))

        Args:
            mask: Binary hazard mask (H, W)
            h, w: Image height, width

        Returns:
            Area-based risk score in [0, 1]
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

        Args:
            depth_val: Modal depth value for hazard (normalized 0-1, lower = closer)

        Returns:
            Depth-based risk score in [0, 1]
        """
        d_ref = self.hyperparams.get("d_ref", 0.5)
        return min(1.0, depth_val / d_ref)

    def generate_risk_map(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        merged_agents: List[Any]
    ) -> Tuple[np.ndarray, List[Hazard]]:
        """
        Constructs the dense spatial risk map C_risk.

        Implements Algorithm 1 Lines 2-8:
        For each hazard/agent:
        1. Get binary mask M_i
        2. Get c_vlm (from hazard metadata or agent)
        3. Get c_conf (detection confidence)
        4. Compute c_area using Eq (3)
        5. Compute depth d_i and c_depth using Eq (4)
        6. Compute total cost C_i = sum(alpha * c)
        7. Update risk_map = max(risk_map, M_i * C_i)

        Args:
            image: The RGB frame (H, W, 3)
            depth_map: The depth map (H, W) normalized [0,1]
            merged_agents: List of DetectedAgent objects with hazard metadata

        Returns:
            risk_map: Dense map of size (H, W) with values in [0, C_max]
            hazards: List of Hazard dataclasses with computed components
        """
        h, w = image.shape[:2]
        risk_map = np.zeros((h, w), dtype=np.float32)
        computed_hazards = []

        # Extract weights
        w_vlm = self.weights.get("vlm", 0.4)
        w_area = self.weights.get("area", 0.2)
        w_conf = self.weights.get("conf", 0.2)
        w_depth = self.weights.get("depth", 0.2)

        for agent in merged_agents:
            # Get mask for front camera (primary view for risk map)
            mask = agent.masks.get("front")
            if mask is None or not np.any(mask):
                # Try other sides
                for side_mask in agent.masks.values():
                    if side_mask is not None and np.any(side_mask):
                        mask = side_mask
                        break

            if mask is None or not np.any(mask):
                continue

            # Ensure mask matches image dimensions
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask = mask > 0.5

            # 1. Contextual VLM Risk (c_vlm)
            # Priority: hazard_metadata.c_vlm > agent.hazard_verified default > 0
            c_vlm = 0.0
            description = ""
            hazard_type = "unknown"

            if hasattr(agent, 'hazard_metadata') and agent.hazard_metadata:
                # Use the highest c_vlm from associated hazards
                c_vlm = max(m.get("c_vlm", 0.0) for m in agent.hazard_metadata)
                # Combine descriptions
                descriptions = [m.get("description", "") for m in agent.hazard_metadata if m.get("description")]
                description = "; ".join(descriptions) if descriptions else f"{agent.class_name} agent"
                hazard_types = [m.get("hazard_type", "unknown") for m in agent.hazard_metadata]
                hazard_type = hazard_types[0] if hazard_types else "unknown"
            else:
                # No hazard metadata - use agent class as fallback
                c_vlm = 0.3  # Default moderate risk for detected agents without VLM scoring
                description = f"{agent.class_name} agent"
                hazard_type = "dynamic_agent"

            # 2. Area-Based Risk (c_area) - Equation (3)
            c_area = self._compute_area_risk(mask, h, w)

            # 3. Confidence Score (c_conf) - Detection confidence
            c_conf = float(agent.seg_score) if hasattr(agent, 'seg_score') else 0.5

            # 4. Depth-Based Risk (c_depth) - Equation (4)
            # Use agent's mode_depth or compute from mask
            if hasattr(agent, 'mode_depth') and agent.mode_depth is not None:
                d_i = agent.mode_depth
            else:
                d_i = mode_depth(depth_map, mask)
            c_depth = self._compute_depth_risk(d_i)

            # Total Scalar Cost C_i = alpha_vlm * c_vlm + alpha_area * c_area + alpha_conf * c_conf + alpha_depth * c_depth
            total_cost = (
                w_vlm * c_vlm +
                w_area * c_area +
                w_conf * c_conf +
                w_depth * c_depth
            )

            # Clamp total cost to reasonable range
            total_cost = max(0.0, min(1.0, total_cost))

            # Update risk map: C_risk(m, n) = max(C_risk(m, n), C_i) for all (m, n) in M_i
            # This implements the pixel-wise max fusion (Algorithm 1 Line 8)
            risk_map = np.maximum(risk_map, mask * total_cost)

            computed_hazards.append(Hazard(
                id=getattr(agent, 'agent_id', 'unknown'),
                mask=mask,
                c_vlm=c_vlm,
                c_conf=c_conf,
                depth=d_i,
                description=description,
                c_area=c_area,
                c_depth=c_depth,
                total_cost=total_cost
            ))

        logger.debug(f"Risk map generated: shape={risk_map.shape}, max={risk_map.max():.3f}, "
                     f"mean={risk_map.mean():.3f}, {len(computed_hazards)} hazards")

        return risk_map, computed_hazards

    def get_risk_statistics(self, risk_map: np.ndarray) -> Dict[str, float]:
        """Get statistics about the risk map."""
        return {
            "min": float(risk_map.min()),
            "max": float(risk_map.max()),
            "mean": float(risk_map.mean()),
            "std": float(risk_map.std()),
            "percentile_90": float(np.percentile(risk_map, 90)),
            "percentile_95": float(np.percentile(risk_map, 95)),
            "high_risk_ratio": float(np.sum(risk_map > 0.5) / risk_map.size),
            "any_risk_ratio": float(np.sum(risk_map > 0.01) / risk_map.size)
        }