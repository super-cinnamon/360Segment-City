# for parallel processing
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union
import os
import pickle
from pathlib import Path
import numpy as np
from src.tasks.preprocessing import (
    load_cubic,
    split_frames,
    generate_cubic,
)
from src.tasks.depth_estimation import (
    predict_depths,
    get_closest_depth_mask,
    predict_cubic_depths,
    mode_depth,
    get_depth_mode,
)
from src.tasks.flow_estimation import (
    predict_flows,
    mode_flow,
    describe_flow,
)
from src.tasks.segmentation import predict_segmentations, predict_cubic_segmentations
from src.tasks.id_tracker import ObjectIDTracker
from src.tasks.environment import query_world_model, unload_world_model
from src.tasks.risk_assessment import (
    RiskAssessmentEngine,
    TelemetryData,
    compute_depth_bounds,
)
from src.tasks.risk_map import RiskMapEngine
from src.tasks.yoloe_segmentation import YOLOESegmenter, create_yoloe_segmenter
from src.tasks.vlm_hazard_identification import VLMHazardIdentifier, create_vlm_hazard_identifier
from src.tasks.hazard_agent_merger import HazardAgentMerger, create_hazard_agent_merger
from src.tasks.hazard_visualization import HazardDetectionLogger, create_hazard_logger, visualize_risk_map
logger = logging.getLogger(__name__)

from src.tasks.roi import (
    build_static_objects_summary,
    resize_frames,
    should_process_window,
)

from src.telemetry import TelemetryProcessor
from src.tasks.config.utils import CONFIG, ENV_PROMPT, render_environment_prompt
from src.tasks.fragment_merger import FragmentMerger
from src.tasks.agent_persistence import RecentAgentHistory, restore_persistent_depth_pruned_agents

class VideoLoader:
    def __init__(self, video_path):
        self.video_path = video_path
        # Do not eagerly load or slice frames here — keep loader lightweight.
        self.frames = None

    def load_cubic(self):  # ! to double check, need cubic generator function
        return load_cubic(self.video_path)
    
    def generate_cubic(self, frames):  
        return generate_cubic(frames)

    def get_split_frames(self, max_to_extract=2000, prune_similar_frames: Optional[bool] = None):  # * if you'd like to only work on a sample, slice this list
        # Backwards-compatible helper that returns up to `max_to_extract` frames
        # starting at the beginning of the video. Pruning is disabled by default.
        if prune_similar_frames is None:
            prune_similar_frames = CONFIG["processing"].get("prune_similar_frames", False)
        self.frames = split_frames(self.video_path, max_to_extract=max_to_extract, prune_similar_frames=prune_similar_frames)
        return self.frames  # [100:200]

    def get_frames_window(self, start_frame: int = 0, max_to_extract: int = 2000, threshold: Optional[float] = None, prune_similar_frames: Optional[bool] = None):
        # New helper: returns (frames_list, last_raw_index) for a window starting
        # at `start_frame`. Uses split_frames_window implemented in preprocessing.
        from src.tasks.preprocessing import split_frames_window
        if threshold is None:
            threshold = CONFIG["processing"].get("frame_similarity_threshold", 2.0)
        if prune_similar_frames is None:
            prune_similar_frames = CONFIG["processing"].get("prune_similar_frames", False)
        frames, last_idx = split_frames_window(
            self.video_path,
            start_frame=start_frame,
            threshold=threshold,
            max_to_extract=max_to_extract,
            prune_similar_frames=prune_similar_frames,
        )
        return frames, last_idx


class VideoProcessor:
    def __init__(self, video_loader: VideoLoader, cubic):
        self.video_loader = video_loader
        self.cubic = cubic
        self.id_tracker = ObjectIDTracker(
            max_missing_frames=int(
                CONFIG.get("agent_persistence", {}).get("max_age_frames", 30)
            )
        )
        self.next_frame_index = 0
        from src.tasks.panorama_tracker import PanoramaTracker
        self.panorama_tracker = PanoramaTracker()
        if cubic:

            # keep this for backward compatibility; code that provides explicit
            # frames will pass them into methods instead
            self.cubic_frames = None

    def _get_cache_path(self, prefix, epoch_idx, raw_start, raw_end):
        if os.environ.get("test") != "1":
            return None

        video_name = os.path.basename(self.video_loader.video_path)
        cache_dir = Path("data/cache/vision")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Use raw_start and raw_end for stable indexing
        if raw_start is not None and raw_end is not None:
            idx_str = f"{raw_start}_{raw_end}"
        elif epoch_idx is not None:
            idx_str = f"epoch_{epoch_idx}"
        else:
            return None

        return cache_dir / f"{prefix}_{video_name}_{idx_str}.pkl"

    def get_depth_mask(self):
        # Backwards-compatible no-arg form: use preloaded frames if present.
        if self.cubic:
            frames = self.cubic_frames
            depths = predict_cubic_depths(frames)
            return depths
        else:
            frames = self.video_loader.frames
            depths = predict_depths(frames)
            return depths

    def get_depth_mask_for(self, frames, epoch_idx=None, raw_start=None, raw_end=None):
        # New API: compute depths for either cubic dict or list of frames
        depth_cache_prefix = "depth_yolo" if get_depth_mode() == "yolo" else "depth"
        cache_path = self._get_cache_path(
            depth_cache_prefix, epoch_idx, raw_start, raw_end
        )
        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        if self.cubic:
            result = predict_cubic_depths(frames)
        else:
            result = predict_depths(frames)

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

        return result

    def get_flow_masks_for(self, frames, epoch_idx=None, raw_start=None, raw_end=None):
        # New API: compute flows for either cubic dict or list of frames
        cache_path = self._get_cache_path("flow", epoch_idx, raw_start, raw_end)
        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        if self.cubic:
            # Process each side
            result = {side: predict_flows(face_frames) for side, face_frames in frames.items()}
        else:
            result = predict_flows(frames)

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

        return result

    def _propagate_hazard_masks(self, agents, prev_frame_idx, curr_frame_idx, flow_maps, all_camera_frames):
        """
        Propagate hazard masks from previous frame to current frame using optical flow.

        Args:
            agents: List of DetectedAgent from previous frame
            prev_frame_idx: Index of previous frame
            curr_frame_idx: Index of current frame
            flow_maps: Dict of {side: list of flow maps} from predict_flows
            all_camera_frames: Dict of {side: list of frames}

        Returns:
            List of DetectedAgent with propagated masks for current frame
        """
        import cv2
        from src.tasks.risk_assessment import DetectedAgent

        if not agents:
            return []

        propagated_agents = []

        for agent in agents:
            # Create a copy of the agent
            new_agent = DetectedAgent(
                agent_id=agent.agent_id,
                class_name=agent.class_name,
                side=agent.side,
                frame=curr_frame_idx,
                seg_score=agent.seg_score,
                mode_depth=agent.mode_depth,
                distance_weight=agent.distance_weight,
                instance_id=agent.instance_id,
                fused_sides=agent.fused_sides.copy(),
                masks={},
                was_fused=agent.was_fused,
                mode_flow=agent.mode_flow,
                flow_description=agent.flow_description
            )

            # Copy hazard metadata
            if hasattr(agent, 'hazard_metadata'):
                new_agent.hazard_metadata = agent.hazard_metadata
            new_agent.hazard_verified = getattr(agent, 'hazard_verified', False)
            new_agent.is_environment_hazard = getattr(agent, 'is_environment_hazard', False)

            # Propagate masks for each side using optical flow
            for side, mask in agent.masks.items():
                if mask is None or not np.any(mask):
                    new_agent.masks[side] = mask
                    continue

                if side in flow_maps and prev_frame_idx < len(flow_maps[side]):
                    flow = flow_maps[side][prev_frame_idx]  # Flow from prev -> curr
                    if flow is not None:
                        # Warp mask using optical flow
                        h, w = mask.shape
                        flow_map = flow[0] if isinstance(flow, tuple) else flow

                        # Create coordinate grids
                        y_coords, x_coords = np.mgrid[0:h, 0:w]

                        # Apply flow to get source coordinates
                        # flow_map[..., 0] = u (x displacement), flow_map[..., 1] = v (y displacement)
                        src_x = x_coords - flow_map[..., 0]
                        src_y = y_coords - flow_map[..., 1]

                        # Clip to valid range
                        src_x = np.clip(src_x, 0, w - 1)
                        src_y = np.clip(src_y, 0, h - 1)

                        # Interpolate mask at source coordinates
                        # Use remap for efficient warping
                        src_x = src_x.astype(np.float32)
                        src_y = src_y.astype(np.float32)
                        warped_mask = cv2.remap(mask.astype(np.float32), src_x, src_y, cv2.INTER_NEAREST)
                        new_agent.masks[side] = warped_mask > 0.5
                    else:
                        new_agent.masks[side] = mask
                else:
                    # No flow available, keep original mask
                    new_agent.masks[side] = mask

            propagated_agents.append(new_agent)

        return propagated_agents

    def segment(self, object_name=None):
        # for this model there is no object name so we ignore for now
        if self.cubic:
            frames = self.cubic_frames
            segmentation_masks = predict_cubic_segmentations(frames)
            return segmentation_masks
        frames = self.video_loader.frames
        segmentation_masks = predict_segmentations(frames)
        return segmentation_masks

    def segment_for(self, frames, object_name=None, epoch_idx=None, raw_start=None, raw_end=None):
        # New API: segmentation for provided frames (either cubic dict or list)
        cache_path = self._get_cache_path("seg", epoch_idx, raw_start, raw_end)
        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        if self.cubic:
            result = predict_cubic_segmentations(frames)
        else:
            result = predict_segmentations(frames)

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

        return result

    def clean_segmentation(self, depth_masks, segmentation_masks):
        # get the closest depth mask for the segmentation mask
        closest_depth_mask = get_closest_depth_mask(depth_masks)

        # clean the segmentation mask using the closest depth mask
        cleaned_segmentation_masks = []
        for seg_mask, depth_mask in zip(segmentation_masks, closest_depth_mask):
            cleaned_mask = seg_mask * depth_mask
            cleaned_segmentation_masks.append(cleaned_mask)
        return cleaned_segmentation_masks
    

class SegmentationPipeline:
    def __init__(self, input_path, cubic=True):
        print("DEBUG: SegmentationPipeline.__init__ start", flush=True)

        # Resolve video path if input_path is a directory (Support for telemetry-enabled runs)
        if os.path.isdir(input_path):
            video_path = os.path.join(input_path, "360_video.mp4")
            logger.info(f"Directory input detected: {input_path}. Initializing TelemetryProcessor.")
            self.telemetry_processor = TelemetryProcessor(input_path)
        else:
            video_path = input_path
            logger.info(f"File input detected: {input_path}. Telemetry disabled (no directory provided).")
            self.telemetry_processor = None

        pipeline_cfg = CONFIG.get("pipeline", {})
        self.split_enabled = pipeline_cfg.get("split_panorama", True)
        self.enabled_views = pipeline_cfg.get("views", ["front", "back", "left", "right"])

        actual_cubic = cubic if self.split_enabled else False
        self.video_loader = VideoLoader(video_path)
        print("DEBUG: VideoLoader initialized", flush=True)
        self.video_processor = VideoProcessor(self.video_loader, cubic=actual_cubic)
        history_cfg = CONFIG.get("agent_persistence", {})
        self.recent_agent_history = RecentAgentHistory(
            max_age_frames=history_cfg.get("max_age_frames", 30)
        )
        print("DEBUG: VideoProcessor initialized", flush=True)
        self.fragment_merger = FragmentMerger()
        print("DEBUG: SegmentationPipeline.__init__ end", flush=True)

    # ! update and fix whatever copilot messed with here and write proper code
    def _get_image_scale(self, image_scale: Optional[float] = None) -> float:
        if image_scale is not None:
            try:
                image_scale = float(image_scale)
            except (TypeError, ValueError):
                image_scale = 0.7
            return max(0.0, min(1.0, image_scale))

        processing_cfg = CONFIG.get("processing", {})
        image_scale = processing_cfg.get("image_scale", 0.7)
        try:
            image_scale = float(image_scale)
        except (TypeError, ValueError):
            image_scale = 0.7
        return max(0.0, min(1.0, image_scale))

    def _prepare_frames_for_inference(self, frames, image_scale: Optional[float] = None):
        scale = self._get_image_scale(image_scale)
        if self.video_processor.cubic:
            if isinstance(frames, dict):
                cubic_frames = frames
            else:
                cubic_frames = self.video_loader.generate_cubic(frames)
            return {
                side: resize_frames(face_frames, scale=scale)
                for side, face_frames in cubic_frames.items()
            }
        return resize_frames(frames, scale=scale)

    def _get_absolute_frame_numbers(self, frame_count: int, raw_start=None) -> list[int]:
        start = (
            int(raw_start)
            if raw_start is not None
            else self.video_processor.next_frame_index
        )
        frame_numbers = list(range(start, start + frame_count))
        self.video_processor.next_frame_index = max(
            self.video_processor.next_frame_index, start + frame_count
        )
        return frame_numbers

    def _attach_recent_history(self, frame_segments, frame_number: int) -> None:
        history_cfg = CONFIG.get("agent_persistence", {})
        score_floor = float(history_cfg.get("score_floor", 0.2))
        score_threshold = float(CONFIG["segmentation"]["score_threshold"])
        for segment in frame_segments:
            history = self.recent_agent_history.match(
                segment, frame_number, score_floor
            )
            if history is None:
                continue
            segment["recent_history"] = history
            if segment.get("score", 0.0) < score_threshold:
                segment["history_rescued"] = True

    def _update_recent_history(self, segmented_items, frame_numbers) -> None:
        score_threshold = float(CONFIG["segmentation"]["score_threshold"])
        for frame_number, frame_segments in zip(frame_numbers, segmented_items):
            self.recent_agent_history.observe_strong_segments(
                frame_segments, frame_number, score_threshold
            )

    def prune_segmentation(
            self,
            items,
            score_threshold=CONFIG["segmentation"]["score_threshold"],
            dynamic_labels=CONFIG["segmentation"]["dynamic_labels"],
            static_labels=CONFIG["segmentation"]["static_labels"]
        ):
        """
        This method aims to prune out all segmentation masks that show a score below the given threshold.
        And also to prune out any label that is irrelevant to our needs. Both are given as input, and will 
        default to the config file defaults.
        This processes a single frame of segmentation items, and returns the pruned list of items.
        """

        items = [
            item for item in items
            if item["class_id"] in dynamic_labels
            and (
                (item.get("score") is not None and item["score"] >= score_threshold)
                or item.get("history_rescued", False)
            )
        ]

        static_items = [item for item in items if item["class_id"] in static_labels]

        return items, static_items

    def prune_depth(self, segmented_items, depth_threshold=CONFIG["segmentation"]["depth_threshold"]):
        min_depth_cfg = CONFIG["segmentation"].get("min_depth", {"sides": 0.1})

        for i, frame_segments in enumerate(segmented_items):
            if not frame_segments:
                continue

            # Sort closest to furthest
            sorted_segments = sorted(frame_segments, key=lambda x: x["mode_depth"])

            # Keep elements that fall within valid depth boundary [min_depth, depth_threshold)
            # Use side-specific min_depth: 0.1 for back, 0.05 for sides
            segmented_items[i] = [
                item for item in sorted_segments
                if item["mode_depth"] is not None and
                min_depth_cfg.get(item["side"], min_depth_cfg.get("sides", 0.1)) <= item["mode_depth"] < depth_threshold
            ]

        return segmented_items

    def process_vision(self, object_name=None, image_scale: Optional[float] = None):
        # Backwards-compatible default behaviour (no frames provided): run as before
        if self.video_loader.frames is None:
            self.video_loader.frames = self.video_loader.get_split_frames()

        frames = self.video_loader.frames
        frame_numbers = self._get_absolute_frame_numbers(len(frames))
        prepared_frames = self._prepare_frames_for_inference(frames, image_scale=image_scale)
        depth_masks = self.video_processor.get_depth_mask_for(prepared_frames)
        segmentation_masks = self.video_processor.segment_for(prepared_frames, object_name)
        flow_masks = self.video_processor.get_flow_masks_for(prepared_frames)

        # PANORAMA TRACKING: Use persistent tracker for continuity across frames and sides
        tracks_per_frame = []
        if not self.video_processor.cubic:
            for frame in frames:
                tracks_per_frame.append(self.video_processor.panorama_tracker.process_frame(frame))

        if not self.video_processor.cubic:
            segmentation_masks = {"front": segmentation_masks}
            depth_masks = {"front": depth_masks}
            flow_masks = {"front": flow_masks}

        # create the list of segmented items with their class names and which frame they belong to
        segmented_items = []
        environment_items = []
        for i in range (len(segmentation_masks["front"])): # looping through frames
            frame_segments = []
            for key in [k for k in segmentation_masks.keys() if k in self.enabled_views]:  # looping through enabled sides
                for segment_info in segmentation_masks[key][i]["segmentation_labels"]:  # loop through segmented items
                    # Retrieve human-readable class name from model's id2label mapping
                    class_name = CONFIG["segmentation"]["id2label"].get(str(segment_info["label_id"]), f"Class_{segment_info['label_id']}")
                    binary_mask = (segmentation_masks[key][i]["segmentation_map"] == segment_info["id"])

                    # Keep IDs consistent across frames using Mask-Matching Cache
                    consistent_id = self.video_processor.id_tracker.get_id(
                        binary_mask, class_name, frame_numbers[i], side=key
                    )

                    # PANORAMA TRACKING: Link segmentation mask to global track ID if available
                    global_track_id = None
                    if not self.video_processor.cubic and tracks_per_frame:
                        # tracks_per_frame is a list of frames, each frame a list of track_data
                        # Find if any global track in this frame overlaps with this mask's centroid
                        coords = np.argwhere(binary_mask)
                        if len(coords) > 0:
                            cy, cx = np.mean(coords, axis=0)
                            for track in tracks_per_frame[i]:
                                tx1, ty1, tx2, ty2 = track['box']
                                if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                                    global_track_id = track['id']
                                    break

                    final_id = global_track_id if global_track_id is not None else consistent_id

                    # calculate the mode of the depth for this object
                    mode_depth_value = mode_depth(depth_masks[key][1][i], binary_mask) # ! this part needs to be switched to take more frames not just current
                    # calculate the average flow for this object
                    mode_flow_value = mode_flow(flow_masks[key][i], binary_mask)
                    frame_segments.append({
                        "frame": i,
                        "side": key,
                        "class_name": class_name,
                        "class_id": segment_info["label_id"],
                        "instance_id": final_id,
                        "agent_identity": f"mask:{key}:{class_name}:{consistent_id}",
                        "score": segment_info.get("score", None),
                        "was_fused": segment_info.get("was_fused", False),
                        "mask": binary_mask,
                        "mode_depth": mode_depth_value,
                        "mode_flow": mode_flow_value,
                        "flow_description": describe_flow(mode_flow_value, side=key),
                    })
            # prune segmentation items based on score and relevant labels
            self._attach_recent_history(frame_segments, frame_numbers[i])
            frame_segments, environment_segments = self.prune_segmentation(frame_segments)
            print(f"DEBUG: Frame {i} - After score prune: {len(frame_segments)} agents")
            segmented_items.append(frame_segments)
            environment_items.append(environment_segments)
        # prune segmentation items based on depth
        if CONFIG.get("fragment_merging", {}).get("enabled", True):
            # Merge occluder-split fragments before depth pruning
            for i in range(len(segmented_items)):
                # Get frame dimensions from the first available depth map
                first_side = list(depth_masks.keys())[0]

                # depth_masks[first_side][1] is a list of depth maps (one per frame)
                # We need the depth map for the current frame index i
                depth_frame = depth_masks[first_side][1][i]
                if depth_frame is None:
                    continue
                h, w = depth_frame.shape[:2]

                for key in depth_masks.keys():
                    side_segments = [s for s in segmented_items[i] if s["side"] == key]
                    if len(side_segments) < 2:
                        continue

                    merged = self.fragment_merger.merge_fragments(
                        side_segments,
                        depth_masks[key][1][i],
                        w, h
                    )
                    if len(merged) < len(side_segments):
                        logger.info(f"Frame {i} [{key}]: Merged fragments. {len(side_segments)} -> {len(merged)}")

                    # Replace old segments with merged ones
                    segmented_items[i] = [s for s in segmented_items[i] if s["side"] != key] + merged

        for i, frame_segments in enumerate(segmented_items):
            segmented_items[i] = self.fragment_merger.merge_side_overlaps(frame_segments)

        candidates_by_frame = [list(frame_segments) for frame_segments in segmented_items]
        segmented_items = self.prune_depth(segmented_items)
        depth_kept_segments = [list(frame_segments) for frame_segments in segmented_items]
        segmented_items = restore_persistent_depth_pruned_agents(
            candidates_by_frame,
            segmented_items,
        )
        self._update_recent_history(depth_kept_segments, frame_numbers)
        print(f"DEBUG: After depth prune - Total frames with agents: {sum(1 for f in segmented_items if f)}")
        environment_items = self.prune_depth(environment_items)

        return segmented_items, environment_items, depth_masks

    # * is the same as the function i wrote above, but just takes specific frames as input, above function will be removed later
    def process_vision_for(self, frames, object_name=None, prepared_frames=None, image_scale: Optional[float] = None, epoch_idx=None, raw_start=None, raw_end=None):
        """
        Process a provided list of frames (or cubic dict) and return segmented items.
        This allows sliding-window processing without changing the core logic.
        """
        if prepared_frames is None:
            prepared_frames = self._prepare_frames_for_inference(frames, image_scale=image_scale)
        frame_count = len(frames["front"]) if isinstance(frames, dict) else len(frames)
        frame_numbers = self._get_absolute_frame_numbers(frame_count, raw_start)

        # compute depth masks and segmentation masks for provided frames
        if self.video_processor.cubic:
            depth_masks = self.video_processor.get_depth_mask_for(
                prepared_frames, epoch_idx=epoch_idx, raw_start=raw_start, raw_end=raw_end
            )
            segmentation_masks = self.video_processor.segment_for(
                prepared_frames, object_name, epoch_idx=epoch_idx, raw_start=raw_start, raw_end=raw_end
            )
            flow_masks = self.video_processor.get_flow_masks_for(
                prepared_frames, epoch_idx=epoch_idx, raw_start=raw_start, raw_end=raw_end
            )
            frames_front = prepared_frames["front"]
        else:
            depth_masks = self.video_processor.get_depth_mask_for(
                prepared_frames, epoch_idx=epoch_idx, raw_start=raw_start, raw_end=raw_end
            )
            segmentation_masks = self.video_processor.segment_for(
                prepared_frames, object_name, epoch_idx=epoch_idx, raw_start=raw_start, raw_end=raw_end
            )
            flow_masks = self.video_processor.get_flow_masks_for(
                prepared_frames, epoch_idx=epoch_idx, raw_start=raw_start, raw_end=raw_end
            )
            frames_front = prepared_frames

        # PANORAMA TRACKING: Use persistent tracker for continuity across frames and sides
        tracks_per_frame = []
        if not self.video_processor.cubic:
            for frame in frames:
                tracks_per_frame.append(self.video_processor.panorama_tracker.process_frame(frame))

        if not self.video_processor.cubic:
            segmentation_masks = {"front": segmentation_masks}
            depth_masks = {"front": depth_masks}
            flow_masks = {"front": flow_masks}


        # create the list of segmented items with their class names and which frame they belong to
        segmented_items = []
        environment_items = []
        for i in range (len(segmentation_masks["front"])): # looping through frames
            frame_segments = []
            for key in [k for k in segmentation_masks.keys() if k in self.enabled_views]:  # looping through enabled sides
                for segment_info in segmentation_masks[key][i]["segmentation_labels"]:  # loop through segmented items
                    # Retrieve human-readable class name from model's id2label mapping
                    class_name = CONFIG["segmentation"]["id2label"].get(str(segment_info["label_id"]), f"Class_{segment_info['label_id']}")
                    binary_mask = (segmentation_masks[key][i]["segmentation_map"] == segment_info["id"])

                    # Keep IDs consistent across frames using Mask-Matching Cache
                    consistent_id = self.video_processor.id_tracker.get_id(
                        binary_mask, class_name, frame_numbers[i], side=key
                    )

                    # PANORAMA TRACKING: Link segmentation mask to global track ID if available
                    global_track_id = None
                    if not self.video_processor.cubic and tracks_per_frame:
                        # tracks_per_frame is a list of frames, each frame a list of track_data
                        # Find if any global track in this frame overlaps with this mask's centroid
                        coords = np.argwhere(binary_mask)
                        if len(coords) > 0:
                            cy, cx = np.mean(coords, axis=0)
                            for track in tracks_per_frame[i]:
                                tx1, ty1, tx2, ty2 = track['box']
                                if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                                    global_track_id = track['id']
                                    break

                    final_id = global_track_id if global_track_id is not None else consistent_id

                    # calculate the mode of the depth for this object
                    mode_depth_value = mode_depth(depth_masks[key][1][i], binary_mask) # ! this part needs to be switched to take more frames not just current
                    # calculate the average flow for this object
                    mode_flow_value = mode_flow(flow_masks[key][i], binary_mask)
                    frame_segments.append({
                        "frame": i,
                        "side": key,
                        "class_name": class_name,
                        "class_id": segment_info["label_id"],
                        "instance_id": final_id,
                        "agent_identity": f"mask:{key}:{class_name}:{consistent_id}",
                        "score": segment_info.get("score", None),
                        "was_fused": segment_info.get("was_fused", False),
                        "mask": binary_mask,
                        "mode_depth": mode_depth_value,
                        "mode_flow": mode_flow_value,
                        "flow_description": describe_flow(mode_flow_value, side=key),
                    })
            # prune segmentation items based on score and relevant labels
            self._attach_recent_history(frame_segments, frame_numbers[i])
            frame_segments, environment_segments = self.prune_segmentation(frame_segments)
            print(f"DEBUG: Frame {i} - After score prune: {len(frame_segments)} agents")
            segmented_items.append(frame_segments)
            environment_items.append(environment_segments)
        # prune segmentation items based on depth
        if CONFIG.get("fragment_merging", {}).get("enabled", True):
            # Merge occluder-split fragments before depth pruning
            for i in range(len(segmented_items)):
                # Get frame dimensions from the first available depth map
                first_side = list(depth_masks.keys())[0]

                # depth_masks[first_side][1] is a list of depth maps (one per frame)
                # We need the depth map for the current frame index i
                depth_frame = depth_masks[first_side][1][i]
                if depth_frame is None:
                    continue
                h, w = depth_frame.shape[:2]

                for key in depth_masks.keys():
                    side_segments = [s for s in segmented_items[i] if s["side"] == key]
                    if len(side_segments) < 2:
                        continue

                    merged = self.fragment_merger.merge_fragments(
                        side_segments,
                        depth_masks[key][1][i],
                        w, h
                    )
                    if len(merged) < len(side_segments):
                        logger.info(f"Frame {i} [{key}]: Merged fragments. {len(side_segments)} -> {len(merged)}")

                    # Replace old segments with merged ones
                    segmented_items[i] = [s for s in segmented_items[i] if s["side"] != key] + merged

        for i, frame_segments in enumerate(segmented_items):
            segmented_items[i] = self.fragment_merger.merge_side_overlaps(frame_segments)

        candidates_by_frame = [list(frame_segments) for frame_segments in segmented_items]
        segmented_items = self.prune_depth(segmented_items)
        depth_kept_segments = [list(frame_segments) for frame_segments in segmented_items]
        segmented_items = restore_persistent_depth_pruned_agents(
            candidates_by_frame,
            segmented_items,
        )
        self._update_recent_history(depth_kept_segments, frame_numbers)
        print(f"DEBUG: After depth prune - Total frames with agents: {sum(1 for f in segmented_items if f)}")
        environment_items = self.prune_depth(environment_items)

        return segmented_items, environment_items, depth_masks

    # * here is where the environment description is generated, the detected static elements will be added here next time
    def process_environment(self, static_objects=None, prompt: str = ENV_PROMPT) -> str:
        # Backwards-compatible behaviour: use preloaded frames if present
        if self.video_processor.cubic and self.video_loader.frames is not None:
            frames = self.video_loader.frames
        elif not self.video_processor.cubic and self.video_loader.frames is not None:
            frames = self.video_loader.frames
        else:
            # No preloaded frames — fall back to loading up to 2000 frames from start
            frames = self.video_loader.get_split_frames()

        return self.process_environment_for(frames, static_objects=static_objects, prompt=prompt)

    def process_environment_for(self, frames, static_objects=None, prompt: str = ENV_PROMPT, prepared_frames=None) -> str:
        if prepared_frames is None:
            prepared_frames = self._prepare_frames_for_inference(frames)

        if self.video_processor.cubic:
            front_frames = prepared_frames["front"]
        else:
            front_frames = prepared_frames

        # add static objects to prompt
        static_objects_summary = static_objects if static_objects is not None else []
        resolved_prompt = render_environment_prompt(
            prompt,
            static_objects=static_objects_summary,
        )

        # Query the world model using all frames in the epoch as a video sequence
        # This provides a comprehensive environment description for ROI gating.
        env_description = query_world_model(
            prompt=resolved_prompt,
            video_frames=front_frames,
            model=CONFIG["vlm"]["world_model"]["model_name"],
        )

        return env_description

    def process_epoch(
            self,
            frames,
            object_name=None,
            telemetry: Optional[TelemetryData] = None,
            prompt: str = ENV_PROMPT,
            roi_enabled: Optional[bool] = None,
            roi_threshold: Optional[float] = None,
            image_scale: Optional[float] = None,
            epoch_idx: Optional[int] = None,
            raw_start: Optional[int] = None,
            raw_end: Optional[int] = None,
            force_process: bool = False,
            viz_output_dir: Optional[str] = None,
    ) -> dict:
        """Run ROI gating, environment description generation, segmentation, and risk assessment for one window."""
        if roi_enabled is None:
            roi_enabled = bool(CONFIG.get("processing", {}).get("roi_enabled", True))
        if roi_threshold is None:
            roi_threshold = float(CONFIG.get("processing", {}).get("roi_threshold", 0.75))

        # Fetch telemetry for this epoch if processor is available
        if telemetry is None and self.telemetry_processor is not None:
            telemetry = self.telemetry_processor.get_epoch_telemetry(epoch_idx if epoch_idx is not None else 0)
            if telemetry:
                logger.debug(f"Epoch {epoch_idx}: Telemetry data successfully injected.")
            else:
                logger.debug(f"Epoch {epoch_idx}: No telemetry data available for this window.")

        prepared_frames = self._prepare_frames_for_inference(frames)
        environment_description = self.process_environment_for(
            frames,
            static_objects=None,
            prompt=prompt,
            prepared_frames=prepared_frames,
        )

        should_process = should_process_window(
            environment_description,
            roi_enabled=roi_enabled,
            roi_threshold=roi_threshold,
        )
        roi_triggered = roi_enabled and should_process
        if not should_process and not force_process:
            return {
                "processed": False,
                "environment_description": environment_description,
                "segmented_items": [],
                "environment_items": [],
                "risk_result": None,
                "reason": "skip_low_roi",
                "roi_triggered": False,
            }

        segmented_items, environment_items, depth_masks = self.process_vision_for(
            frames,
            object_name=object_name,
            prepared_frames=prepared_frames,
            epoch_idx=epoch_idx,
            raw_start=raw_start,
            raw_end=raw_end,
        )

        # Build agents from segmented items for hazard merging
        from src.tasks.risk_assessment import build_agents_from_segments, compute_depth_bounds
        d_min, d_max = compute_depth_bounds(segmented_items)
        segmented_agents = [
            build_agents_from_segments(f, d_min, d_max) for f in segmented_items
        ]

        # --- TWO-STAGE HAZARD DETECTION PIPELINE (per paper Algorithm 1) ---
        # Stage 1: VLM Hazard Identification (with motorcycle state)
        # Stage 2: YOLOE Segmentation for identified hazards
        # Stage 3: Merge hazards with segmented agents

        # Get front frame for hazard detection (first frame of epoch)
        front_frame = None
        if self.video_processor.cubic:
            front_frames = prepared_frames.get("front", [])
        else:
            front_frames = prepared_frames

        if front_frames and len(front_frames) > 0:
            front_frame = front_frames[0]
        elif frames and isinstance(frames, dict):
            front_frame = frames.get("front", [None])[0] if frames.get("front") else None
        elif frames and isinstance(frames, list) and len(frames) > 0:
            front_frame = frames[0]

        merged_agents = segmented_agents[0] if segmented_agents else []  # Use first frame's agents

        # Get ALL camera frames for the epoch (for video generation)
        camera_sides = ["front", "left", "right", "back"]
        all_camera_frames = {}
        all_depth_maps = {}

        if self.video_processor.cubic:
            for side in camera_sides:
                if side in prepared_frames and prepared_frames[side]:
                    all_camera_frames[side] = prepared_frames[side]
                if side in depth_masks and depth_masks[side] and len(depth_masks[side]) > 1:
                    all_depth_maps[side] = depth_masks[side][1]  # List of depth maps per frame
        else:
            # Non-cubic: assume single camera (front)
            if prepared_frames and isinstance(prepared_frames, list):
                all_camera_frames["front"] = prepared_frames
            if depth_masks and "front" in depth_masks:
                all_depth_maps["front"] = depth_masks["front"][1]

        # Also check original frames
        if not all_camera_frames and frames and isinstance(frames, dict):
            for side in camera_sides:
                if side in frames and frames[side]:
                    all_camera_frames[side] = frames[side]
        elif not all_camera_frames and frames and isinstance(frames, list):
            all_camera_frames["front"] = frames

        # Initialize hazard detection logger for this epoch
        hazard_logger = create_hazard_logger(epoch_idx=epoch_idx)

        # Check if we have any frames
        has_frames = any(len(f) > 0 for f in all_camera_frames.values())

        if has_frames and (merged_agents or True):  # Always run hazard detection
            try:
                # Get motorcycle state from telemetry
                speed_kmh = telemetry.speed_kmh if telemetry else 0.0
                lean_angle_deg = telemetry.roll if telemetry else 0.0

                # Use first available frame (prefer front) for VLM hazard identification
                first_frame = None
                first_depth_map = None
                for side in ["front", "left", "right", "back"]:
                    if side in all_camera_frames and all_camera_frames[side]:
                        first_frame = all_camera_frames[side][0]
                        if side in all_depth_maps and all_depth_maps[side]:
                            first_depth_map = all_depth_maps[side][0]
                        break

                if first_frame is None:
                    raise ValueError("No frames available for hazard detection")

                if first_depth_map is None:
                    first_depth_map = np.ones(first_frame.shape[:2], dtype=np.float32) * 0.5

                # Stage 1: VLM Hazard Identification (on first frame with motorcycle state)
                vlm_identifier = create_vlm_hazard_identifier()
                hazards = vlm_identifier.identify_hazards(first_frame, speed_kmh, lean_angle_deg)

                # Log VLM input (prompt and motorcycle state)
                vlm_prompt = vlm_identifier._build_prompt(speed_kmh, lean_angle_deg) if hasattr(vlm_identifier, '_build_prompt') else "N/A"
                hazard_logger.log_vlm_input(first_frame, vlm_prompt, speed_kmh, lean_angle_deg)

                # Log VLM output (raw response and parsed hazards)
                vlm_raw_response = getattr(vlm_identifier, 'last_raw_response', 'N/A')
                hazard_logger.log_vlm_output(vlm_raw_response, hazards)

                # Stage 2: YOLOE Segmentation for identified hazards on ALL camera sides (first frame)
                hazard_labels = [h["label"] for h in hazards] if hazards else []
                yoloe_results_per_side = {}

                if hazard_labels:
                    yoloe_segmenter = create_yoloe_segmenter()

                    for side in camera_sides:
                        if side in all_camera_frames and all_camera_frames[side]:
                            side_first_frame = all_camera_frames[side][0]
                            side_results = yoloe_segmenter.segment_hazards(side_first_frame, hazard_labels)

                            # Combine VLM c_vlm with YOLOE results
                            for h_vlm, h_yoloe in zip(hazards, side_results):
                                h_yoloe["c_vlm"] = h_vlm.get("c_vlm", 0.0)
                                h_yoloe["hazard_type"] = h_vlm.get("hazard_type", "environment_hazard")
                                h_yoloe["description"] = h_vlm.get("description", "")

                            yoloe_results_per_side[side] = side_results

                            # Log YOLOE results for this side
                            hazard_logger.log_yoloe_results(side_first_frame, hazard_labels, side_results)

                # Stage 3: Merge hazards with segmented agents for each side
                if merged_agents:
                    merger = create_hazard_agent_merger()

                    # For each side, merge hazards with agents that have masks on that side
                    for side in camera_sides:
                        if side in yoloe_results_per_side:
                            side_yoloe_results = yoloe_results_per_side[side]
                            side_depth_map = all_depth_maps[side][0] if side in all_depth_maps and all_depth_maps[side] else first_depth_map

                            # Filter agents that have masks on this side
                            side_agents = [a for a in merged_agents if side in a.masks and a.masks[side] is not None and np.any(a.masks[side])]

                            if side_agents:
                                # Create side-specific merged agents
                                merged_side_agents = merger.merge(side_agents, side_yoloe_results, side_depth_map)

                                # Update the original agents with merged masks for this side
                                for merged_agent in merged_side_agents:
                                    # Find corresponding agent in merged_agents
                                    for orig_agent in merged_agents:
                                        if orig_agent.agent_id == merged_agent.agent_id:
                                            # Update mask for this side
                                            for s, mask in merged_agent.masks.items():
                                                if mask is not None and np.any(mask):
                                                    orig_agent.masks[s] = mask
                                            # Update hazard metadata
                                            if hasattr(merged_agent, 'hazard_metadata'):
                                                orig_agent.hazard_metadata = merged_agent.hazard_metadata
                                            orig_agent.hazard_verified = getattr(merged_agent, 'hazard_verified', False)
                                            orig_agent.is_environment_hazard = getattr(merged_agent, 'is_environment_hazard', False)
                                            break
                            else:
                                # No existing agents on this side - create new hazard agents
                                from src.tasks.risk_assessment import DetectedAgent
                                from src.tasks.depth_estimation import mode_depth

                                for h in side_yoloe_results:
                                    hazard_mask = h["mask"]
                                    try:
                                        hazard_depth = mode_depth(side_depth_map, hazard_mask)
                                    except Exception:
                                        hazard_depth = 0.5

                                    hazard_label = h["label"].replace(" ", "_").replace("-", "_")
                                    agent_id = f"hazard_{hazard_label}_{side}_{len(merged_agents)}"
                                    distance_weight = 1.0 - hazard_depth

                                    new_agent = DetectedAgent(
                                        agent_id=agent_id,
                                        class_name=h["label"],
                                        side=side,
                                        frame=0,
                                        seg_score=min(1.0, h.get("confidence", 0.5) * 1.2),
                                        mode_depth=hazard_depth,
                                        distance_weight=distance_weight,
                                        instance_id=-1,
                                        fused_sides=[side],
                                        masks={side: hazard_mask},
                                        was_fused=False,
                                        mode_flow=None,
                                        flow_description=None
                                    )
                                    new_agent.hazard_metadata = [{
                                        "label": h["label"],
                                        "c_vlm": h.get("c_vlm", 0.0),
                                        "hazard_type": h.get("hazard_type", "environment_hazard"),
                                        "description": h.get("description", ""),
                                        "hazard_confidence": h.get("confidence", 0.0),
                                        "verified": True
                                    }]
                                    new_agent.hazard_verified = True
                                    new_agent.is_environment_hazard = True
                                    merged_agents.append(new_agent)
                else:
                    # No existing agents at all - create from hazards on all sides
                    from src.tasks.risk_assessment import DetectedAgent
                    from src.tasks.depth_estimation import mode_depth

                    for side in camera_sides:
                        if side in yoloe_results_per_side:
                            side_yoloe_results = yoloe_results_per_side[side]
                            side_depth_map = all_depth_maps[side][0] if side in all_depth_maps and all_depth_maps[side] else first_depth_map

                            for h in side_yoloe_results:
                                hazard_mask = h["mask"]
                                try:
                                    hazard_depth = mode_depth(side_depth_map, hazard_mask)
                                except Exception:
                                    hazard_depth = 0.5

                                hazard_label = h["label"].replace(" ", "_").replace("-", "_")
                                agent_id = f"hazard_{hazard_label}_{side}_{len(merged_agents)}"
                                distance_weight = 1.0 - hazard_depth

                                new_agent = DetectedAgent(
                                    agent_id=agent_id,
                                    class_name=h["label"],
                                    side=side,
                                    frame=0,
                                    seg_score=min(1.0, h.get("confidence", 0.5) * 1.2),
                                    mode_depth=hazard_depth,
                                    distance_weight=distance_weight,
                                    instance_id=-1,
                                    fused_sides=[side],
                                    masks={side: hazard_mask},
                                    was_fused=False,
                                    mode_flow=None,
                                    flow_description=None
                                )
                                new_agent.hazard_metadata = [{
                                    "label": h["label"],
                                    "c_vlm": h.get("c_vlm", 0.0),
                                    "hazard_type": h.get("hazard_type", "environment_hazard"),
                                    "description": h.get("description", ""),
                                    "hazard_confidence": h.get("confidence", 0.0),
                                    "verified": True
                                }]
                                new_agent.hazard_verified = True
                                new_agent.is_environment_hazard = True
                                merged_agents.append(new_agent)

                # Log merged agents for first frame (front camera)
                merger_info = {"num_yoloe_hazards": sum(len(r) for r in yoloe_results_per_side.values()), "num_merged": len(merged_agents)}
                hazard_logger.log_merged_agents(first_frame, first_depth_map, segmented_agents[0] if segmented_agents else [],
                    [r for results in yoloe_results_per_side.values() for r in results], merged_agents, merger_info)

                # Stage 4: Propagate hazard masks across frames using optical flow
                # Get flow estimates between consecutive frames for each side
                flow_maps = {}
                if self.video_processor.cubic and hasattr(self, 'flow_maps') and self.flow_maps:
                    # Use pre-computed flow maps if available
                    flow_maps = self.flow_maps
                else:
                    # Compute flow between consecutive frames for each side
                    from src.tasks.flow_estimation import predict_flows
                    for side in camera_sides:
                        if side in all_camera_frames and len(all_camera_frames[side]) > 1:
                            side_frames = all_camera_frames[side]
                            try:
                                side_flows = predict_flows(side_frames)
                                flow_maps[side] = side_flows  # List of flow maps (frame_i -> frame_i+1)
                            except Exception as e:
                                logger.warning(f"Failed to compute flow for {side}: {e}")

                # Stage 5: Generate per-frame visualizations for ALL frames on ALL sides
                risk_map_engine = RiskMapEngine()
                num_frames = max(len(f) for f in all_camera_frames.values()) if all_camera_frames else 0

                # Track per-frame merged agents (propagated masks)
                frame_merged_agents = [None] * num_frames
                frame_merged_agents[0] = merged_agents  # First frame uses merged agents from Stage 3

                # Propagate masks to subsequent frames using optical flow
                for frame_idx in range(1, num_frames):
                    # Propagate masks from previous frame using optical flow
                    propagated_agents = self.video_processor._propagate_hazard_masks(frame_merged_agents[frame_idx - 1], frame_idx - 1, frame_idx, flow_maps, all_camera_frames)
                    frame_merged_agents[frame_idx] = propagated_agents

                # Now generate visualizations for all frames
                for frame_idx in range(num_frames):
                    # Get frames for all sides at this frame index
                    side_frames = {}
                    side_depth_maps = {}
                    for side in camera_sides:
                        if side in all_camera_frames and frame_idx < len(all_camera_frames[side]):
                            side_frames[side] = all_camera_frames[side][frame_idx]
                        if side in all_depth_maps and frame_idx < len(all_depth_maps[side]):
                            side_depth_maps[side] = all_depth_maps[side][frame_idx]
                        elif side in all_depth_maps and all_depth_maps[side]:
                            side_depth_maps[side] = all_depth_maps[side][0]

                    # Use front frame as primary for logging
                    primary_frame = side_frames.get("front", list(side_frames.values())[0] if side_frames else None)
                    primary_depth = side_depth_maps.get("front", list(side_depth_maps.values())[0] if side_depth_maps else None)

                    if primary_frame is None:
                        continue

                    # Get merged agents for this frame
                    current_agents = frame_merged_agents[frame_idx] if frame_merged_agents[frame_idx] else merged_agents

                    # Generate risk map for primary (front) camera
                    risk_map, hazard_objects = risk_map_engine.generate_risk_map(primary_frame, primary_depth, current_agents)

                    # Create composite visualization showing all camera sides
                    side_visualizations = []
                    for side in camera_sides:
                        if side in side_frames:
                            side_frame = side_frames[side]
                            side_depth = side_depth_maps.get(side, primary_depth)

                            # Filter agents that have masks on this side
                            side_agents = [a for a in current_agents if side in a.masks and a.masks[side] is not None and np.any(a.masks[side])]

                            # Create merged agent visualization for this side
                            merged_viz = hazard_logger.create_merged_viz(side_frame, side_depth, side_agents)

                            # Create risk map visualization for this side
                            if side_agents:
                                side_risk_map, _ = risk_map_engine.generate_risk_map(side_frame, side_depth, side_agents)
                                risk_viz = visualize_risk_map(side_risk_map, side_frame,
                                    str(hazard_logger.session_dir / f"risk_map_{side}_{frame_idx:04d}.jpg"))
                            else:
                                risk_viz = visualize_risk_map(np.zeros_like(side_depth), side_frame,
                                    str(hazard_logger.session_dir / f"risk_map_{side}_{frame_idx:04d}.jpg"))

                            # Combine side by side
                            combined_side = np.hstack([merged_viz, risk_viz])
                            side_visualizations.append(combined_side)

                    # Combine all sides horizontally
                    if side_visualizations:
                        combined_viz = np.hstack(side_visualizations)
                    else:
                        combined_viz = np.zeros((480, 1280, 3), dtype=np.uint8)

                    # Log per-frame merged agents (for first few frames)
                    if frame_idx < 5:
                        hazard_logger.frame_counter = frame_idx
                        hazard_logger.log_merged_agents(primary_frame, primary_depth, [],
                            [r for results in yoloe_results_per_side.values() for r in results], current_agents,
                            {"frame_idx": frame_idx, "num_merged": len(current_agents)})

                    # Add to video
                    hazard_logger.add_frame_to_video(combined_viz)

                logger.info(f"Hazard detection complete: {len(hazards)} VLM hazards, {len(merged_agents)} merged agents, {num_frames} frames visualized across {len(camera_sides)} sides")

            except Exception as e:
                logger.warning(f"Hazard detection pipeline failed: {e}")
                import traceback
                traceback.print_exc()
                merged_agents = segmented_agents[0] if segmented_agents else []

        # Finalize hazard detection video
        hazard_logger.finalize_video()

        static_objects = build_static_objects_summary(environment_items)
        refined_environment_description = self.process_environment_for(
            frames,
            static_objects=static_objects,
            prompt=prompt,
            prepared_frames=prepared_frames,
        )

        risk_result = self.process_risk(
            segmented_items,
            refined_environment_description,
            frames=prepared_frames,
            depth_masks=depth_masks,
            telemetry=telemetry,
            epoch_idx=epoch_idx,
            raw_start_frame=raw_start or 0,
            viz_output_dir=viz_output_dir,
            merged_agents=merged_agents,
        )

        return {
            "processed": True,
            "environment_description": refined_environment_description,
            "segmented_items": segmented_items,
            "environment_items": environment_items,
            "risk_result": risk_result,
            "reason": "processed",
            "roi_triggered": roi_triggered,
        }

    # * the full risk processing pipeline, using sliding window of 60 frames (assuming framerate of gopro is 60)
    def process_risk(
            self,
            segmented_items: list[list[dict]],
            env_description: str,
            frames:          any = None,
            depth_masks:     any = None,
            telemetry: Optional[TelemetryData] = None,
            api_base: str = CONFIG["risk_assessment"]["api_base"],
            model_name: Optional[str] = None,
            epoch_idx: Optional[int] = None,
            raw_start_frame: int = 0,
            viz_output_dir: Optional[str] = None,
            merged_agents: Optional[list] = None,
        ) -> Union[list[dict], dict]:
            """
            Runs G-VEval risk assessment across the batch of frames as an "epoch" window.

            Args:
                segmented_items:  Output of SegmentationPipeline.process_vision().
                                A list of frames, where each frame is a list of
                                segment dicts (class_name, mode_depth, mask, …).
                env_description:  Structured environment description string produced
                                by query_world_model() / query_ollama_vlm() in
                                environment.py.
                telemetry:        Optional TelemetryData from physics/IMU sensors.
                                Pass None (default) when no sensor data is available.
                api_base:         Base URL of the vLLM-compatible API server.
                model_name:       Name of the model hosted by the API server.
                epoch_window_size: Optional window size if evaluating multiple sub-epochs.
                                If None (default), the entire batch is evaluated as a single epoch window.

            Returns:
                Epoch G-VEval result dict (or list of dicts if sub-epochs are requested).
            """
            model_name = model_name or CONFIG["risk_assessment"]["model_name"]
            engine = RiskAssessmentEngine(
                api_base=api_base,
                model_name=model_name,
                n_reasons=int(CONFIG["risk_assessment"].get("n_reasons", 5)),
            )

            risk_result = engine.assess_epoch(
                segmented_items=segmented_items,
                env_description=env_description,
                frames=frames,
                depth_maps=depth_masks,
                telemetry=telemetry,
                epoch_id=f"epoch_{epoch_idx}" if epoch_idx is not None else "latest",
                merged_agents=merged_agents,
            )

            # Generate Hazard Visualization Video
            try:
                from src.tasks.risk_assessment import build_agents_from_segments, compute_depth_bounds, generate_hazard_viz_video
                d_min, d_max = compute_depth_bounds(segmented_items)
                epoch_agents = [
                    build_agents_from_segments(f, d_min, d_max) for f in segmented_items
                ]

                # Extract per-frame telemetry for the epoch if available
                epoch_telemetry = None
                if self.telemetry_processor and epoch_idx is not None:
                    df = self.telemetry_processor.data
                    start = epoch_idx * 30
                    end = start + 30
                    if start < len(df):
                        epoch_df = df.iloc[start:end]
                        epoch_telemetry = [
                            TelemetryData(
                                ttc=0.0, pet=0.0, d_lat=0.0, v_rel=0.0,
                                speed_ms=float(row['speed_ms']) if 'speed_ms' in row else 0.0,
                                speed_kmh=float(row['speed_kmh']) if 'speed_kmh' in row else 0.0,
                                a_ego=float(row['a_x']) if 'a_x' in row else 0.0,
                                roll=float(row['roll']) if 'roll' in row else 0.0,
                                yaw_rate=float(row['gy_z']) if 'gy_z' in row else 0.0,
                            )
                            for _, row in epoch_df.iterrows()
                        ]

                viz_dir = Path(viz_output_dir or "outputs/risk_viz")
                viz_path = str(viz_dir / (
                    f"hazard_viz_{epoch_idx}.mp4"
                    if epoch_idx is not None
                    else f"hazard_viz_{os.getpid()}.mp4"
                ))
                generate_hazard_viz_video(
                    output_path=viz_path,
                    frames=frames,
                    epoch_agents=epoch_agents,
                    reasons_detail=risk_result.get("reasons", []),
                    epoch_telemetry=epoch_telemetry,
                    raw_start_frame=raw_start_frame,
                )
                risk_result["viz_video_path"] = viz_path
            except Exception as e:
                logger.warning("Failed to generate hazard visualization video: %s", e)

            return risk_result

    def cleanup_gpu(self):
        """Explicitly clear CUDA cache and perform garbage collection to prevent VRAM leaks."""
        from src.tasks import depth_estimation, segmentation, flow_estimation
        depth_estimation.unload_model()
        segmentation.unload_model()
        flow_estimation.unload_model()

        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
