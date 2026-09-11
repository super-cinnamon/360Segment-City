# for parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import os
import pickle
from pathlib import Path
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
)
from src.tasks.segmentation import predict_segmentations, predict_cubic_segmentations
from src.tasks.environment import query_world_model
from src.tasks.risk_assessment import (
    RiskAssessmentEngine,
    TelemetryData,
    compute_depth_bounds,
)
from src.tasks.roi import (
    build_static_objects_summary,
    resize_frames,
    should_process_window,
)

from src.tasks.config.utils import CONFIG, ENV_PROMPT, render_environment_prompt

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
        cache_path = self._get_cache_path("depth", epoch_idx, raw_start, raw_end)
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
        # Ablation: depth cleaning disabled
        return segmentation_masks
    

class SegmentationPipeline:
    def __init__(self, video_path, cubic=True):
        print("DEBUG: SegmentationPipeline.__init__ start", flush=True)
        self.video_loader = VideoLoader(video_path)
        print("DEBUG: VideoLoader initialized", flush=True)
        self.video_processor = VideoProcessor(self.video_loader, cubic=cubic)
        print("DEBUG: VideoProcessor initialized", flush=True)
        print("DEBUG: SegmentationPipeline.__init__ end", flush=True)

    # ! update and fix whatever copilot messed with here and write proper code
    def _get_image_scale(self, image_scale: float | None = None) -> float:
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

    def _prepare_frames_for_inference(self, frames, image_scale: float | None = None):
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
            if item["score"] >= score_threshold and item["class_id"] in dynamic_labels
        ]

        static_items = [item for item in items if item["class_id"] in static_labels]

        return items, static_items

    def prune_depth(self, segmented_items, depth_threshold=CONFIG["segmentation"]["depth_threshold"]):
        # Ablation: depth pruning disabled
        return segmented_items

    def process_vision(self, object_name=None, image_scale: float | None = None):
        # Backwards-compatible default behaviour (no frames provided): run as before
        if self.video_loader.frames is None:
            self.video_loader.frames = self.video_loader.get_split_frames()

        prepared_frames = self._prepare_frames_for_inference(self.video_loader.frames, image_scale=image_scale)
        depth_masks = self.video_processor.get_depth_mask_for(prepared_frames)
        segmentation_masks = self.video_processor.segment_for(prepared_frames, object_name)
        # create the list of segmented items with their class names and which frame they belong to
        segmented_items = []
        environment_items = []
        for i in range (len(segmentation_masks["front"])): # looping through frames
            frame_segments = []
            for key in segmentation_masks.keys():  # looping through sides
                for segment_info in segmentation_masks[key][i]["segmentation_labels"]:  # loop through segmented items
                    # Retrieve human-readable class name from model's id2label mapping
                    class_name = CONFIG["segmentation"]["id2label"].get(str(segment_info["label_id"]), f"Class_{segment_info['label_id']}")
                    binary_mask = (segmentation_masks[key][i]["segmentation_map"] == segment_info["id"])
                    # calculate the mode of the depth for this object
                    mode_depth_value = mode_depth(depth_masks[key][1][i], binary_mask) # ! this part needs to be switched to take more frames not just current
                    frame_segments.append({
                        "frame": i,
                        "side": key,
                        "class_name": class_name,
                        "class_id": segment_info["label_id"],
                        "score": segment_info.get("score", None),
                        "was_fused": segment_info.get("was_fused", False),
                        "mask": binary_mask,
                        "mode_depth": mode_depth_value,
                    })
            # prune segmentation items based on score and relevant labels
            frame_segments, environment_segments = self.prune_segmentation(frame_segments)
            segmented_items.append(frame_segments)
            environment_items.append(environment_segments)
        # prune segmentation items based on depth
        # Ablation: disabled
        pass
        pass
      
        return segmented_items, environment_items

    # * is the same as the function i wrote above, but just takes specific frames as input, above function will be removed later
    def process_vision_for(self, frames, object_name=None, prepared_frames=None, image_scale: float | None = None, epoch_idx=None, raw_start=None, raw_end=None):
        """
        Process a provided list of frames (or cubic dict) and return segmented items.
        This allows sliding-window processing without changing the core logic.
        """
        if prepared_frames is None:
            prepared_frames = self._prepare_frames_for_inference(frames, image_scale=image_scale)

        # compute depth masks and segmentation masks for provided frames
        if self.video_processor.cubic:
            depth_masks = self.video_processor.get_depth_mask_for(
                prepared_frames, epoch_idx=epoch_idx, raw_start=raw_start, raw_end=raw_end
            )
            segmentation_masks = self.video_processor.segment_for(
                prepared_frames, object_name, epoch_idx=epoch_idx, raw_start=raw_start, raw_end=raw_end
            )
            frames_front = prepared_frames["front"]
        else:
            depth_masks = self.video_processor.get_depth_mask_for(
                prepared_frames, epoch_idx=epoch_idx, raw_start=raw_start, raw_end=raw_end
            )
            segmentation_masks = self.video_processor.segment_for(
                prepared_frames, object_name, epoch_idx=epoch_idx, raw_start=raw_start, raw_end=raw_end
            )
            frames_front = prepared_frames

        # create the list of segmented items with their class names and which frame they belong to
        segmented_items = []
        environment_items = []
        for i in range (len(segmentation_masks["front"])): # looping through frames
            frame_segments = []
            for key in segmentation_masks.keys():  # looping through sides
                for segment_info in segmentation_masks[key][i]["segmentation_labels"]:  # loop through segmented items
                    # Retrieve human-readable class name from model's id2label mapping
                    class_name = CONFIG["segmentation"]["id2label"].get(str(segment_info["label_id"]), f"Class_{segment_info['label_id']}")
                    binary_mask = (segmentation_masks[key][i]["segmentation_map"] == segment_info["id"])
                    # calculate the mode of the depth for this object
                    mode_depth_value = mode_depth(depth_masks[key][1][i], binary_mask) # ! this part needs to be switched to take more frames not just current
                    frame_segments.append({
                        "frame": i,
                        "side": key,
                        "class_name": class_name,
                        "class_id": segment_info["label_id"],
                        "score": segment_info.get("score", None),
                        "was_fused": segment_info.get("was_fused", False),
                        "mask": binary_mask,
                        "mode_depth": mode_depth_value,
                    })
            # prune segmentation items based on score and relevant labels
            frame_segments, environment_segments = self.prune_segmentation(frame_segments)
            segmented_items.append(frame_segments)
            environment_items.append(environment_segments)
        # prune segmentation items based on depth
        # Ablation: disabled
        pass
        pass

        return segmented_items, environment_items

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
            images=front_frames,
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
    ) -> dict:
        """Run ROI gating, environment description generation, segmentation, and risk assessment for one window."""
        if roi_enabled is None:
            roi_enabled = bool(CONFIG.get("processing", {}).get("roi_enabled", True))
        if roi_threshold is None:
            roi_threshold = float(CONFIG.get("processing", {}).get("roi_threshold", 0.75))

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
        if not should_process:
            return {
                "processed": False,
                "environment_description": environment_description,
                "segmented_items": [],
                "environment_items": [],
                "risk_result": None,
                "reason": "skip_low_roi",
            }

        segmented_items, environment_items = self.process_vision_for(
            frames,
            object_name=object_name,
            prepared_frames=prepared_frames,
            epoch_idx=epoch_idx,
            raw_start=raw_start,
            raw_end=raw_end,
        )
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
            telemetry=telemetry,
        )

        return {
            "processed": True,
            "environment_description": refined_environment_description,
            "segmented_items": segmented_items,
            "environment_items": environment_items,
            "risk_result": risk_result,
            "reason": "processed",
        }

    # * the full risk processing pipeline, using sliding window of 60 frames (assuming framerate of gopro is 60)
    def process_risk(
            self,
            segmented_items: list[list[dict]],
            env_description: str,
            frames:          any = None,
            telemetry: Optional[TelemetryData] = None,
            api_base: str = CONFIG["risk_assessment"]["api_base"],
            model_name: str = CONFIG["risk_assessment"]["model_name"],
        ) -> list[dict] | dict:
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
            engine = RiskAssessmentEngine(
                api_base=api_base,
                model_name=model_name,
                n_reasons=int(CONFIG["risk_assessment"].get("n_reasons", 5)),
            )

            return engine.assess_epoch(
                segmented_items=segmented_items,
                env_description=env_description,
                frames=frames,
                telemetry=telemetry,
            )

    def cleanup_gpu(self):
        """Explicitly clear CUDA cache and perform garbage collection to prevent VRAM leaks."""
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

