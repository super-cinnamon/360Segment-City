
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DramaAdapter:
    """
    Adapter that extracts only the whitelisted information from the pipeline output.
    Handles data quality checks and recomputes cleaned scores.
    """

    def __init__(self, extract_text_quality: bool = False):
        self.extract_text_quality = extract_text_quality
        # Leaked CoT markers
        self.cot_markers = ["Okay, let's", "First, ", "I see", "As an AI", "Let me think"]

    def _is_valid_reason(self, reason: Dict[str, Any], segmented_items: List[List[Dict[str, Any]]]) -> bool:
        """
        Checks if a reason is valid based on the whitelist constraints.
        """
        # 1. Score probabilities check
        score_probs = reason.get("score_probs", {})
        if not score_probs or sum(score_probs.values()) == 0:
            return False

        # 2. Expected score check
        if reason.get("expected_score", 0) == 0:
            return False

        # 3. CoT leakage check
        danger_reasoning = reason.get("danger_reasoning", "")
        if any(marker in danger_reasoning for marker in self.cot_markers):
            return False

        # 4. Object existence check
        # Get the object_id of the first hazard in the reason
        hazards = reason.get("hazards", [])
        if not hazards:
            return False

        oid = hazards[0].get("object_id")
        # Search for this oid in the segmented items for this epoch
        # Note: segmented_items is a list of frames, each frame is a list of segments
        found = False
        for frame in segmented_items:
            for seg in frame:
                if seg.get("instance_id") == oid or f"agent_{seg.get('instance_id')}" == oid:
                    found = True
                    break
            if found: break

        if not found:
            return False

        return True

    def _get_bbox_from_mask(self, mask: np.ndarray) -> List[int]:
        """Derives a 2D bounding box from a binary mask."""
        if mask is None or not np.any(mask):
            return [0, 0, 0, 0]
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return [x0, y0, x1, y1]

    def extract(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts a compact record from the pipeline output.
        """
        if not pipeline_output.get("processed", False):
            return {
                "clip_id": pipeline_output.get("clip_id"),
                "processed": False,
                "skip_reason": pipeline_output.get("reason")
            }

        risk_res = pipeline_output.get("risk_result", {})
        segmented_items = pipeline_output.get("segmented_items", [])

        # Extract general scores
        extracted = {
            "clip_id": pipeline_output.get("clip_id"),
            "processed": True,
            "general_risk_score": risk_res.get("general_risk_score"),
            "score_probabilities": risk_res.get("score_probabilities"),
            "reasons": []
        }

        # Process reasons
        raw_reasons = risk_res.get("reasons", [])
        valid_reasons = []

        for r in raw_reasons:
            is_valid = self._is_valid_reason(r, segmented_items)

            # Compact reason record
            compact_r = {
                "softmax_weight": r.get("softmax_weight"),
                "expected_score": r.get("expected_score"),
                "score_probs": r.get("score_probs"),
                "valid": is_valid,
            }

            if is_valid:
                # Extract hazard info
                hazards = r.get("hazards", [])
                if hazards:
                    h = hazards[0]
                    compact_r["object_id"] = h.get("object_id")
                    compact_r["hazard_type"] = h.get("hazard_type")

                    # Derive bbox from mask
                    # Note: we need the mask from the pipeline's segmented_items
                    # matching the object_id and the frame
                    bbox = None
                    for frame in segmented_items:
                        for seg in frame:
                            if f"agent_{seg.get('instance_id')}" == h.get("object_id") or seg.get("instance_id") == h.get("object_id"):
                                bbox = self._get_bbox_from_mask(seg.get("mask"))
                                break
                        if bbox: break
                    compact_r["bbox"] = bbox

                # Optional text quality fields
                if self.extract_text_quality:
                    if hazards:
                        h = hazards[0]
                        compact_r["observation"] = h.get("observation")
                        compact_r["danger_reasoning"] = h.get("danger_reasoning")
                        compact_r["actionable_risk"] = h.get("actionable_risk")

            extracted["reasons"].append(compact_r)
            if is_valid:
                valid_reasons.append(r)

        # Recompute benchmark score from valid reasons
        if valid_reasons:
            weights = [r.get("softmax_weight", 0.0) for r in valid_reasons]
            sum_w = sum(weights)
            if sum_w > 0:
                norm_weights = [w / sum_w for w in weights]
                cleaned_score = sum(w * r.get("expected_score", 0) for w, r in zip(norm_weights, valid_reasons))
                extracted["cleaned_risk_score"] = cleaned_score
            else:
                extracted["cleaned_risk_score"] = None
        else:
            extracted["cleaned_risk_score"] = None

        return extracted
