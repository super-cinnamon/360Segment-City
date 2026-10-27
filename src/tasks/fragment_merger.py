import numpy as np
import cv2
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from src.tasks.config.utils import CONFIG

logger = logging.getLogger(__name__)

class FragmentMerger:
    """
    Post-processing step to merge nearby panoptic instance fragments.
    """
    def __init__(self):
        self.cfg = CONFIG.get("fragment_merging", {})
        self._face_projection_cache = {}

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

    def _get_face_projection(self, side: str, height: int, width: int, overlap: float):
        cache_key = (side, height, width, overlap)
        if cache_key in self._face_projection_cache:
            return self._face_projection_cache[cache_key]

        u_axis = np.linspace(-1.0 - overlap, 1.0 + overlap, width)
        if side in ("right", "left"):
            u_axis = u_axis[::-1]
        v_axis = np.linspace(1.0 + overlap, -1.0 - overlap, height)
        uu, vv = np.meshgrid(u_axis, v_axis)

        if side == "front":
            theta = np.arctan2(uu, np.ones_like(uu))
        elif side == "right":
            theta = np.arctan2(np.ones_like(uu), uu)
        elif side == "back":
            theta = np.arctan2(-uu, -np.ones_like(uu))
        elif side == "left":
            theta = np.arctan2(-np.ones_like(uu), -uu)
        else:
            raise ValueError(f"Unknown cube side: {side}")

        phi = np.arctan2(vv, np.sqrt(uu**2 + vv**2 + 1.0))
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(phi)
        z = np.cos(theta) * np.cos(phi)

        pano_height = max(2 * height, 1)
        pano_width = max(4 * width, 1)
        longitude = (theta / (2.0 * np.pi) + 0.5) * pano_width
        latitude = (np.pi / 2.0 - phi) / np.pi * pano_height
        pano_x = np.rint(longitude).astype(np.int64) % pano_width
        pano_y = np.clip(np.rint(latitude).astype(np.int64), 0, pano_height - 1)
        projection = (
            (pano_y * pano_width + pano_x).ravel(),
            x.ravel(),
            y.ravel(),
            z.ravel(),
        )
        self._face_projection_cache[cache_key] = projection
        return projection

    @staticmethod
    def _visible_on_side(x, y, z, side: str, overlap: float):
        horizontal = np.sqrt(x**2 + z**2)
        v = y / np.where(horizontal > 0, horizontal, 1.0)
        if side == "front":
            valid = z > 0
            denom = np.where(valid, z, 1.0)
            u = x / denom
        elif side == "right":
            valid = x > 0
            denom = np.where(valid, x, 1.0)
            u = z / denom
        elif side == "back":
            valid = z < 0
            denom = np.where(valid, z, -1.0)
            u = x / denom
        elif side == "left":
            valid = x < 0
            denom = np.where(valid, x, -1.0)
            u = z / denom
        else:
            return np.zeros_like(x, dtype=bool)

        limit = 1.0 + overlap
        return valid & (np.abs(u) <= limit) & (np.abs(v) <= limit)

    def merge_side_overlaps(self, frame_segments: List[Dict[str, Any]], overlap: Optional[float] = None):
        """Give same-class detections on different faces one shared instance ID."""
        if len(frame_segments) < 2:
            return frame_segments

        processing_cfg = CONFIG.get("processing", {})
        if overlap is None:
            overlap = float(processing_cfg.get("side_overlap", 0.0))
        if overlap <= 0:
            return frame_segments

        match_cfg = processing_cfg.get("side_overlap_matching", {})
        min_coverage = float(match_cfg.get("min_coverage", 0.2))
        min_pixels = int(match_cfg.get("min_pixels", 4))
        mergeable_classes = set(self.cfg.get("mergeable_classes", []))
        candidates = [
            (index, segment)
            for index, segment in enumerate(frame_segments)
            if segment.get("class_name") in mergeable_classes
            and segment.get("side") in ("front", "right", "back", "left")
            and isinstance(segment.get("mask"), np.ndarray)
        ]
        if len(candidates) < 2:
            return frame_segments

        parent = list(range(len(candidates)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        projected_masks = []
        for _, segment in candidates:
            mask = np.asarray(segment["mask"], dtype=bool)
            height, width = mask.shape
            pano_indices, x, y, z = self._get_face_projection(
                segment["side"], height, width, overlap
            )
            active = mask.ravel()
            projected_masks.append((
                segment["side"],
                segment["class_name"],
                pano_indices,
                x,
                y,
                z,
                active,
            ))

        for left in range(len(candidates)):
            side_a, class_a, pano_a, x_a, y_a, z_a, active_a = projected_masks[left]
            for right in range(left + 1, len(candidates)):
                side_b, class_b, pano_b, x_b, y_b, z_b, active_b = projected_masks[right]
                if side_a == side_b or class_a != class_b:
                    continue

                common_a = active_a & self._visible_on_side(x_a, y_a, z_a, side_b, overlap)
                common_b = active_b & self._visible_on_side(x_b, y_b, z_b, side_a, overlap)
                indices_a = np.unique(pano_a[common_a])
                indices_b = np.unique(pano_b[common_b])
                if min(indices_a.size, indices_b.size) < min_pixels:
                    continue

                shared_pixels = np.intersect1d(indices_a, indices_b, assume_unique=True).size
                coverage = shared_pixels / min(indices_a.size, indices_b.size)
                if coverage >= min_coverage:
                    union(left, right)

        groups = {}
        for index in range(len(candidates)):
            groups.setdefault(find(index), []).append(index)

        for group in groups.values():
            if len(group) < 2:
                continue
            group_segments = [candidates[index][1] for index in group]
            shared_segment = min(group_segments, key=lambda segment: segment["instance_id"])
            shared_id = shared_segment["instance_id"]
            for segment in group_segments:
                segment["instance_id"] = shared_id
                if "agent_identity" in shared_segment:
                    segment["agent_identity"] = shared_segment["agent_identity"]
                segment["was_fused"] = True

        return frame_segments

    def merge_fragments(self, frame_segments: List[Dict[str, Any]], depth_map: np.ndarray, frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
        """
        Merge nearby same-class fragments when their geometry and depth agree.

        Args:
            frame_segments: List of instance dicts for the current frame.
            depth_map: Depth map for the current frame.
            frame_width: Width of the frame.
            frame_height: Height of the frame.

        Returns:
            Updated list of frame segments.
        """
        if not self.cfg.get("enabled", True) or len(frame_segments) < 2:
            return frame_segments

        mergeable_classes = self.cfg.get("mergeable_classes", [])
        max_gap = self.cfg.get("max_gap_px", 40)
        depth_tol = self.cfg.get("depth_rel_tol", 0.10)
        align_min = self.cfg.get("vertical_overlap_min", 0.6)
        min_score = self.cfg.get("min_score", 0.7)
        weights = self.cfg.get("weights", {"proximity": 0.5, "depth_consistency": 0.3, "boundary_alignment": 0.2})

        # 1. Candidate Selection
        candidate_indices = [
            index for index, segment in enumerate(frame_segments)
            if segment["class_name"] in mergeable_classes
        ]
        candidates = [frame_segments[index] for index in candidate_indices]
        if len(candidates) < 2:
            return frame_segments

        # Precompute BBoxes for candidates
        for s in candidates:
            s["_bbox"] = self._get_bbox(s["mask"])

        # 2. Pairwise Analysis
        adj = {i: [] for i in range(len(candidates))}
        pairs_processed = 0

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                A, B = candidates[i], candidates[j]

                # Same class hard gate
                if A["class_name"] != B["class_name"]:
                    continue

                # Fast BBox Gap Check
                ax1, ay1, ax2, ay2 = A["_bbox"]
                bx1, by1, bx2, by2 = B["_bbox"]

                gap_x = max(0, bx1 - ax2 - 1, ax1 - bx2 - 1)
                gap_y = max(0, by1 - ay2 - 1, ay1 - by2 - 1)
                gap_distance = math.hypot(gap_x, gap_y)
                if gap_distance > max_gap:
                    continue

                # Measure depth near each fragment's boundary facing the other.
                kernel_size = max(1, 2 * max_gap + 1)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                dilA = cv2.dilate(A["mask"].astype(np.uint8), kernel, iterations=1)
                dilB = cv2.dilate(B["mask"].astype(np.uint8), kernel, iterations=1)

                def get_boundary_depth(mask, other_dilated):
                    boundary_mask = (mask > 0) & (other_dilated > 0)
                    boundary_pixels = depth_map[boundary_mask]
                    boundary_pixels = boundary_pixels[
                        np.isfinite(boundary_pixels) & (boundary_pixels > 0)
                    ]
                    if boundary_pixels.size == 0:
                        return None
                    return np.median(boundary_pixels)

                dA = get_boundary_depth(A["mask"], dilB)
                dB = get_boundary_depth(B["mask"], dilA)

                score_depth = 0.0
                if dA is not None and dB is not None:
                    relative_depth_difference = abs(dA - dB) / max(abs(dA), abs(dB), 1e-6)
                    if relative_depth_difference > depth_tol:
                        continue
                    score_depth = 1.0
                else:
                    continue

                # Boundary Alignment
                cA = [(ax1 + ax2) / 2, (ay1 + ay2) / 2]
                cB = [(bx1 + bx2) / 2, (by1 + by2) / 2]
                dx, dy = abs(cA[0] - cB[0]), abs(cA[1] - cB[1])

                if gap_x > gap_y or (gap_x == gap_y and dx > dy):
                    score_align = self._calculate_range_iou((ay1, ay2), (by1, by2))
                else:
                    score_align = self._calculate_range_iou((ax1, ax2), (bx1, bx2))

                if score_align < align_min:
                    continue

                proximity_score = 1.0 - gap_distance / max(max_gap, 1)
                total_score = (weights["proximity"] * proximity_score +
                               weights["depth_consistency"] * score_depth +
                               weights["boundary_alignment"] * score_align)

                if total_score >= min_score:
                    adj[i].append(j)
                    adj[j].append(i)
                    logger.debug(f"Candidate pair ({A['instance_id']}, {B['instance_id']}) passed merge check: score={total_score:.3f}, proximity={proximity_score:.3f}, depth={score_depth:.3f}, align={score_align:.3f}")
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
                score = candidates[idx].get("score")
                weighted_conf += (score if score is not None else 0.6) * area

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
                    "agent_identity": candidates[largest_idx].get("agent_identity"),
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

        # Now create the result list
        result_segments = list(frame_segments)

        # Remove all merged fragments
        # We iterate backwards to not mess up indices
        to_remove = sorted((candidate_indices[i] for i in merged_indices), reverse=True)
        for idx in to_remove:
            result_segments.pop(idx)

        # Add the merged instances
        for _, merged_inst in final_merges:
            result_segments.append(merged_inst)

        logger.debug(f"Merged {len(merged_groups)} groups into {len(final_merges)} instances. Pairs processed: {pairs_processed}")
        return result_segments
