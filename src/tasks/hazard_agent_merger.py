"""
Hazard-Agent Merger Module
==========================

Merges YOLOE-detected hazards with OneFormer segmentation agents.
Implements the user's requirement: perfect matches merge masks and boost
priority/risk/confidence; environment hazards become new agents.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from src.tasks.config.utils import CONFIG
from src.tasks.risk_assessment import DetectedAgent

logger = logging.getLogger(__name__)


@dataclass
class MergedAgentInfo:
    """Information about how a hazard was merged with an agent."""
    hazard_label: str
    hazard_c_vlm: float
    hazard_type: str
    hazard_confidence: float
    iou: float
    match_type: str  # "perfect", "partial", "new"
    original_agent_id: Optional[str] = None


class HazardAgentMerger:
    """
    Merges YOLOE hazards with existing segmentation agents.

    Matching logic:
    - IoU > merge_threshold (0.9): PERFECT MATCH → union masks, boost confidence
    - IoU > associate_threshold (0.5): PARTIAL MATCH → associate metadata only
    - IoU <= associate_threshold: NEW ENVIRONMENT HAZARD → create new agent
    """

    def __init__(
        self,
        iou_merge_threshold: float = None,
        iou_associate_threshold: float = None,
        risk_boost_factor: float = None
    ):
        self.iou_merge_threshold = iou_merge_threshold or CONFIG["yoloe"].get("iou_threshold_merge", 0.9)
        self.iou_associate_threshold = iou_associate_threshold or CONFIG["yoloe"].get("iou_threshold_associate", 0.5)
        self.risk_boost_factor = risk_boost_factor or CONFIG["yoloe"].get("risk_boost_factor", 1.2)

        logger.info(f"HazardAgentMerger: merge_thresh={self.iou_merge_threshold}, "
                    f"associate_thresh={self.iou_associate_threshold}, "
                    f"boost_factor={self.risk_boost_factor}")

    def merge(
        self,
        segmented_agents: List[DetectedAgent],
        yoloe_hazards: List[Dict],
        depth_map: np.ndarray
    ) -> List[DetectedAgent]:
        """
        Merge YOLOE hazards with OneFormer segmentation agents.

        Args:
            segmented_agents: List of DetectedAgent from OneFormer segmentation
            yoloe_hazards: List of dicts from YOLOE with keys:
                - label, mask (H,W bool), confidence, bbox, class_id
                - c_vlm (from VLM), hazard_type, description
            depth_map: Depth map for computing hazard depth

        Returns:
            Updated list of DetectedAgent with merged/associated hazards
        """
        if not segmented_agents and not yoloe_hazards:
            return []

        # Create a working copy of agents (we'll modify these)
        merged_agents = []
        for agent in segmented_agents:
            # Create a copy to avoid modifying originals
            merged_agents.append(self._copy_agent(agent))

        # Track merge info for each agent
        merge_info: Dict[str, List[MergedAgentInfo]] = {agent.agent_id: [] for agent in merged_agents}

        # Process each YOLOE hazard
        for hazard in yoloe_hazards:
            hazard_mask = hazard.get("mask")
            if hazard_mask is None or not np.any(hazard_mask):
                continue

            # Find best matching agent across all sides
            best_match = self._find_best_match(hazard_mask, merged_agents)

            if best_match is not None:
                best_agent, best_iou, best_side = best_match

                if best_iou >= self.iou_merge_threshold:
                    # PERFECT MATCH: Union masks, boost confidence
                    self._merge_perfect_match(
                        best_agent, best_side, hazard, best_iou, merge_info
                    )
                elif best_iou >= self.iou_associate_threshold:
                    # PARTIAL MATCH: Associate metadata only
                    self._associate_partial_match(
                        best_agent, best_side, hazard, best_iou, merge_info
                    )
                else:
                    # NEW HAZARD: Create new agent
                    self._create_new_hazard_agent(
                        hazard, merged_agents, depth_map, merge_info
                    )
            else:
                # No agents at all - create new hazard agent
                self._create_new_hazard_agent(
                    hazard, merged_agents, depth_map, merge_info
                )

        # Apply risk boost to merged agents
        self._apply_risk_boost(merged_agents, merge_info)

        logger.info(f"Merged {len(yoloe_hazards)} hazards with {len(segmented_agents)} agents → {len(merged_agents)} total")
        return merged_agents

    def _find_best_match(
        self,
        hazard_mask: np.ndarray,
        agents: List[DetectedAgent]
    ) -> Optional[Tuple[DetectedAgent, float, str]]:
        """Find agent with highest IoU to hazard mask across all sides."""
        best_iou = 0.0
        best_agent = None
        best_side = None

        for agent in agents:
            for side, agent_mask in agent.masks.items():
                if agent_mask is None or not np.any(agent_mask):
                    continue

                iou = self._compute_iou(hazard_mask, agent_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_agent = agent
                    best_side = side

        if best_agent is not None and best_iou > 0:
            return best_agent, best_iou, best_side
        return None

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        # Ensure same shape
        if mask1.shape != mask2.shape:
            # Resize mask2 to match mask1
            import cv2
            mask2 = cv2.resize(mask2.astype(np.uint8),
                             (mask1.shape[1], mask1.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
            mask2 = mask2 > 0.5

        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0
        return intersection / union

    def _merge_perfect_match(
        self,
        agent: DetectedAgent,
        side: str,
        hazard: Dict,
        iou: float,
        merge_info: Dict[str, List[MergedAgentInfo]]
    ):
        """Merge hazard mask into agent (union), boost confidence."""
        hazard_mask = hazard["mask"]

        # Union the masks
        if side in agent.masks and agent.masks[side] is not None:
            agent.masks[side] = np.logical_or(agent.masks[side], hazard_mask)
        else:
            agent.masks[side] = hazard_mask.copy()

        # Boost confidence
        old_conf = agent.seg_score
        new_conf = min(1.0, old_conf * self.risk_boost_factor)
        agent.seg_score = new_conf

        # Add hazard metadata
        if not hasattr(agent, 'hazard_metadata'):
            agent.hazard_metadata = []

        agent.hazard_metadata.append({
            "label": hazard["label"],
            "c_vlm": hazard.get("c_vlm", 0.0),
            "hazard_type": hazard.get("hazard_type", "environment_hazard"),
            "description": hazard.get("description", ""),
            "hazard_confidence": hazard.get("confidence", 0.0),
            "verified": True
        })

        # Mark as hazard verified
        agent.hazard_verified = True

        # Record merge info
        merge_info[agent.agent_id].append(MergedAgentInfo(
            hazard_label=hazard["label"],
            hazard_c_vlm=hazard.get("c_vlm", 0.0),
            hazard_type=hazard.get("hazard_type", "environment_hazard"),
            hazard_confidence=hazard.get("confidence", 0.0),
            iou=iou,
            match_type="perfect",
            original_agent_id=agent.agent_id
        ))

        logger.debug(f"PERFECT MATCH: {hazard['label']} → {agent.agent_id} (IoU={iou:.3f}, conf {old_conf:.2f}→{new_conf:.2f})")

    def _associate_partial_match(
        self,
        agent: DetectedAgent,
        side: str,
        hazard: Dict,
        iou: float,
        merge_info: Dict[str, List[MergedAgentInfo]]
    ):
        """Associate hazard metadata with agent without merging masks."""
        # Add hazard metadata
        if not hasattr(agent, 'hazard_metadata'):
            agent.hazard_metadata = []

        agent.hazard_metadata.append({
            "label": hazard["label"],
            "c_vlm": hazard.get("c_vlm", 0.0),
            "hazard_type": hazard.get("hazard_type", "environment_hazard"),
            "description": hazard.get("description", ""),
            "hazard_confidence": hazard.get("confidence", 0.0),
            "verified": True,
            "partial_match": True
        })

        agent.hazard_verified = True

        merge_info[agent.agent_id].append(MergedAgentInfo(
            hazard_label=hazard["label"],
            hazard_c_vlm=hazard.get("c_vlm", 0.0),
            hazard_type=hazard.get("hazard_type", "environment_hazard"),
            hazard_confidence=hazard.get("confidence", 0.0),
            iou=iou,
            match_type="partial",
            original_agent_id=agent.agent_id
        ))

        logger.debug(f"PARTIAL MATCH: {hazard['label']} → {agent.agent_id} (IoU={iou:.3f})")

    def _create_new_hazard_agent(
        self,
        hazard: Dict,
        merged_agents: List[DetectedAgent],
        depth_map: np.ndarray,
        merge_info: Dict[str, List[MergedAgentInfo]]
    ):
        """Create new DetectedAgent for environment hazard."""
        from src.tasks.depth_estimation import mode_depth
        from src.tasks.risk_assessment import DetectedAgent

        hazard_mask = hazard["mask"]

        # Compute depth for this hazard
        try:
            hazard_depth = mode_depth(depth_map, hazard_mask)
        except Exception:
            hazard_depth = 0.5  # default

        # Create agent ID
        hazard_label = hazard["label"].replace(" ", "_").replace("-", "_")
        agent_id = f"hazard_{hazard_label}_{len(merged_agents)}"

        # Compute distance weight (same as build_agents_from_segments)
        # We'll use a default depth range for hazards
        distance_weight = 1.0 - hazard_depth  # closer = higher weight

        # Create new agent
        new_agent = DetectedAgent(
            agent_id=agent_id,
            class_name=hazard["label"],
            side="front",  # hazards detected on front camera
            frame=0,
            seg_score=min(1.0, hazard.get("confidence", 0.5) * self.risk_boost_factor),
            mode_depth=hazard_depth,
            distance_weight=distance_weight,
            instance_id=-1,  # negative for hazards
            fused_sides=["front"],
            masks={"front": hazard_mask},
            was_fused=False,
            mode_flow=None,
            flow_description=None
        )

        # Add hazard metadata
        new_agent.hazard_metadata = [{
            "label": hazard["label"],
            "c_vlm": hazard.get("c_vlm", 0.0),
            "hazard_type": hazard.get("hazard_type", "environment_hazard"),
            "description": hazard.get("description", ""),
            "hazard_confidence": hazard.get("confidence", 0.0),
            "verified": True
        }]
        new_agent.hazard_verified = True
        new_agent.is_environment_hazard = True

        merged_agents.append(new_agent)
        merge_info[agent_id] = [MergedAgentInfo(
            hazard_label=hazard["label"],
            hazard_c_vlm=hazard.get("c_vlm", 0.0),
            hazard_type=hazard.get("hazard_type", "environment_hazard"),
            hazard_confidence=hazard.get("confidence", 0.0),
            iou=0.0,
            match_type="new",
            original_agent_id=None
        )]

        logger.debug(f"NEW HAZARD AGENT: {hazard['label']} → {agent_id} (conf={new_agent.seg_score:.2f})")

    def _apply_risk_boost(self, agents: List[DetectedAgent], merge_info: Dict[str, List[MergedAgentInfo]]):
        """Apply risk boost factor to agents that have verified hazards."""
        for agent in agents:
            info_list = merge_info.get(agent.agent_id, [])
            has_perfect_match = any(m.match_type == "perfect" for m in info_list)
            has_any_match = len(info_list) > 0

            if has_perfect_match:
                # Already boosted in _merge_perfect_match, but ensure minimum
                agent.seg_score = max(agent.seg_score, 0.7)
            elif has_any_match:
                # Partial match gets smaller boost
                agent.seg_score = min(1.0, agent.seg_score * 1.1)

    def _copy_agent(self, agent: DetectedAgent) -> DetectedAgent:
        """Create a copy of a DetectedAgent."""
        from src.tasks.risk_assessment import DetectedAgent

        return DetectedAgent(
            agent_id=agent.agent_id,
            class_name=agent.class_name,
            side=agent.side,
            frame=agent.frame,
            seg_score=agent.seg_score,
            mode_depth=agent.mode_depth,
            distance_weight=agent.distance_weight,
            instance_id=agent.instance_id,
            fused_sides=agent.fused_sides.copy(),
            masks={k: v.copy() if v is not None else None for k, v in agent.masks.items()},
            was_fused=agent.was_fused,
            mode_flow=agent.mode_flow,
            flow_description=agent.flow_description
        )


def create_hazard_agent_merger(
    iou_merge_threshold: float = None,
    iou_associate_threshold: float = None,
    risk_boost_factor: float = None
) -> HazardAgentMerger:
    """Factory function to create HazardAgentMerger."""
    return HazardAgentMerger(
        iou_merge_threshold=iou_merge_threshold,
        iou_associate_threshold=iou_associate_threshold,
        risk_boost_factor=risk_boost_factor
    )