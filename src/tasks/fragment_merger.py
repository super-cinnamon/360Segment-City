import numpy as np
import cv2
import logging
from typing import List, Dict, Any, Optional, Tuple
from src.tasks.config.utils import CONFIG

logger = logging.getLogger(__name__)

class FragmentMerger:
    """
    Post-processing step to merge panoptic instance fragments split by ego-vehicle occluders.
    """
    def __init__(self):
        self.cfg = CONFIG.get("fragment_merging", {})
        if not self.cfg.get("enabled", True):
            self.ego_mask = None
            return

        self.ego_mask_path = self.cfg.get("ego_mask_path")
        self.ego_mask = self._load_ego_mask()

    def _load_ego_mask(self) -> Optional[np.ndarray]:
        """Loads the static ego-vehicle mask and resizes it to the pipeline scale."""
        if not self.ego_mask_path:
            logger.error("FragmentMerger: ego_mask_path not provided in config.")
            return None

        try:
            mask = cv2.imread(self.ego_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read ego mask at {self.ego_mask_path}")

            # Binarize
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = (mask > 0).astype(np.uint8)

            # Note: We resize the mask in merge_fragments if the frame size changes,
            # but typically the pipeline resolution is fixed by image_scale.
            return mask
        except Exception as e:
            logger.error(f"FragmentMerger: Failed to load ego mask: {e}")
            return None

    def _get_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Returns (x1, y1, x2, y2) for a binary mask."""
        coords = np.argwhere(mask)
        if coords.size == 0:
            return (0, 0, 0, 0)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return (int(x_min), int(y_min), int(x_max), int(y_max))

    def _calculate_range_iou(self, r1: Tuple[int, int], r2: Tuple[int, int]) -> float:
        """Calculates IoU of two 1D ranges."""
        start = max(r1[0], r2[0])
        end = min(r1[1], r2[1])
        if start >= end:
            return 0.0
        intersection = end - start
        union = (r1[1] - r1[0]) + (r2[1] - r2[0]) - intersection
        return intersection / union if union > 0 else 0.0

    def merge_fragments(self, frame_segments: List[Dict[str, Any]], depth_map: np.ndarray, frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
        """
        Merges fragments of the same class split by the ego-vehicle mask.

        Args:
            frame_segments: List of instance dicts for the current frame.
            depth_map: Depth map for the current frame.
            frame_width: Width of the frame.
            frame_height: Height of the frame.

        Returns:
            Updated list of frame segments.
        """
        if not self.cfg.get("enabled", True) or self.ego_mask is None or len(frame_segments) < 2:
            return frame_segments

        # Resize ego mask to current frame size if necessary
        current_ego_mask = cv2.resize(self.ego_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        mergeable_classes = self.cfg.get("mergeable_classes", [])
        max_gap = self.cfg.get("max_gap_px", 40)
        ego_min = self.cfg.get("ego_overlap_min", 0.5)
        depth_tol = self.cfg.get("depth_rel_tol", 0.10)
        align_min = self.cfg.get("vertical_overlap_min", 0.6)
        min_score = self.cfg.get("min_score", 0.7)
        weights = self.cfg.get("weights", {"ego_occlusion": 0.5, "depth_consistency": 0.3, "boundary_alignment": 0.2})

        # 1. Candidate Selection
        candidates = [s for s in frame_segments if s["class_name"] in mergeable_classes]
        if len(candidates) < 2:
            return frame_segments

        # Precompute BBoxes for candidates
        for s in candidates:
            s["_bbox"] = self._get_bbox(s["mask"])

        # 2. Pairwise Analysis
        adj = {i: [] for i in range(len(candidates))}
        pairs_processed = 0
        merges_count = 0

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                A, B = candidates[i], candidates[j]

                # Same class hard gate
                if A["class_name"] != B["class_name"]:
                    continue

                # Fast BBox Gap Check
                ax1, ay1, ax2, ay2 = A["_bbox"]
                bx1, by1, bx2, by2 = B["_bbox"]

                # Check if dilated BBoxes overlap
                if not (ax1 - max_gap <= bx2 + max_gap and bx1 - max_gap <= ax2 + max_gap and
                        ay1 - max_gap <= by2 + max_gap and by1 - max_gap <= ay2 + max_gap):
                    continue

                # Ego-Occluder Explanation
                # Gap region G = (dilate(A) & dilate(B)) \ (A | B)
                kernel = np.ones((max_gap, max_gap), np.uint8)
                dilA = cv2.dilate(A["mask"].astype(np.uint8), kernel, iterations=1)
                dilB = cv2.dilate(B["mask"].astype(np.uint8), kernel, iterations=1)

                gap_region = (cv2.bitwise_and(dilA, dilB) > 0) & ~((A["mask"] | B["mask"]))
                gap_area = np.sum(gap_region)

                if gap_area == 0:
                    # If they touch or are too far, we skip (per requirements: "if the masks directly touch, skip")
                    continue

                ego_overlap = np.sum(gap_region & (current_ego_mask > 0)) / gap_area
                if ego_overlap < ego_min:
                    continue

                # Depth Consistency
                # Use only pixels at the boundary (near the gap) for robust depth comparison
                def get_boundary_depth(mask, other_dilated):
                    boundary_mask = (mask > 0) & (other_dilated > 0)
                    boundary_pixels = depth_map[boundary_mask]
                    if boundary_pixels.size == 0:
                        return None
                    return np.median(boundary_pixels)

                dA = get_boundary_depth(A["mask"], dilB)
                dB = get_boundary_depth(B["mask"], dilA)

                score_depth = 0.0
                if dA is not None and dB is not None:
                    if abs(dA - dB) / max(dA, dB, 1e-6) <= depth_tol:
                        score_depth = 1.0

                # Boundary Alignment
                # Determine split direction based on center distance
                cA = [(ax1 + ax2) / 2, (ay1 + ay2) / 2]
                cB = [(bx1 + bx2) / 2, (by1 + by2) / 2]
                dx, dy = abs(cA[0] - cB[0]), abs(cA[1] - cB[1])

                if dx > dy: # Vertical split (A and B are left/right)
                    score_align = self._calculate_range_iou((ay1, ay2), (by1, by2))
                else: # Horizontal split (A and B are top/bottom)
                    score_align = self._calculate_range_iou((ax1, ax2), (bx1, bx2))

                if score_align < align_min:
                    continue

                # Final Score
                total_score = (weights["ego_occlusion"] * ego_overlap +
                               weights["depth_consistency"] * score_depth +
                               weights["boundary_alignment"] * score_align)

                if total_score >= min_score:
                    adj[i].append(j)
                    adj[j].append(i)
                    merges_count += 1
                    logger.debug(f"Candidate pair ({A['instance_id']}, {B['instance_id']}) passed merge check: score={total_score:.3f}, ego={ego_overlap:.3f}, depth={score_depth:.3f}, align={score_align:.3f}")
                else:
                    if total_score > 0.3: # Only log near-misses to avoid spam
                        logger.debug(f"Candidate pair ({A['instance_id']}, {B['instance_id']}) failed merge: score={total_score:.3f} < {min_score}")

                pairs_processed += 1

        # 3. Grouping (Connected Components)
        visited = set()
        merged_groups = []
        for i in range(len(candidates)):
            if i not in visited:
                component = []
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        component.append(node)
                        stack.extend(adj[node])
                if len(component) > 1:
                    merged_groups.append(component)

        # 4. Merge Execution
        if not merged_groups:
            return frame_segments

        # Create a mapping from old candidates index to new merged instance
        # To avoid over-merging, we check aspect ratio of the union BBox
        final_merges = []
        for group in merged_groups:
            union_mask = np.zeros_like(candidates[0]["mask"], dtype=bool)
            total_area = 0
            weighted_conf = 0.0

            for idx in group:
                m = candidates[idx]["mask"]
                area = np.sum(m)
                union_mask |= m
                total_area += area
                weighted_conf += (m["score"] if "score" in candidates[idx] else 0.6) * area

            # Aspect Ratio check
            ux1, uy1, ux2, uy2 = self._get_bbox(union_mask)
            uw, uh = ux2 - ux1, uy2 - uy1
            aspect_ratio = max(uw, uh) / max(min(uw, uh), 1e-6)

            if aspect_ratio <= self.cfg.get("max_merged_aspect_ratio", 3.0):
                # Use ID of the largest member
                largest_idx = max(group, key=lambda idx: np.sum(candidates[idx]["mask"]))

                merged_instance = {
                    "frame": candidates[0]["frame"],
                    "side": candidates[0]["side"],
                    "class_name": candidates[0]["class_name"],
                    "class_id": candidates[0]["class_id"],
                    "instance_id": candidates[largest_idx]["instance_id"],
                    "score": weighted_conf / total_area if total_area > 0 else 0.6,
                    "was_fused": True,
                    "mask": union_mask,
                    "mode_depth": np.median(depth_map[union_mask]) if np.any(union_mask) else None,
                    "mode_flow": candidates[largest_idx].get("mode_flow"),
                    "flow_description": candidates[largest_idx].get("flow_description"),
                }
                final_merges.append((group, merged_instance))
            else:
                logger.debug(f"Merged group rejected due to aspect ratio: {aspect_ratio:.2f}")

        # Build final result list
        # 1. Identify all indices that were merged
        merged_indices = set()
        for group, _ in final_merges:
            merged_indices.update(group)

        # 2. Reconstruct the frame_segments list
        # We must preserve original order and those not merged.
        # Since candidates is a subset of frame_segments, we need to map indices.

        # Map candidate index back to frame_segments index
        cand_to_orig = []
        for s in frame_segments:
            if s["class_name"] in mergeable_classes:
                cand_to_orig.append(frame_segments.index(s))

        # Now create the result list
        result_segments = list(frame_segments)

        # Remove all merged fragments
        # We iterate backwards to not mess up indices
        to_remove = sorted([cand_to_orig[i] for i in merged_indices], reverse=True)
        for idx in to_remove:
            result_segments.pop(idx)

        # Add the merged instances
        for _, merged_inst in final_merges:
            result_segments.append(merged_inst)

        logger.debug(f"Merged {len(merged_groups)} groups into {len(final_merges)} instances. Pairs processed: {pairs_processed}")
        return result_segments
