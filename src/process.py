# for parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
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

from src.tasks.config.utils import CONFIG, ENV_PROMPT

class VideoLoader:
    def __init__(self, video_path):
        self.video_path = video_path
        # Do not eagerly load or slice frames here — keep loader lightweight.
        self.frames = None

    def load_cubic(self):  # ! to double check, need cubic generator function
        return load_cubic(self.video_path)
    
    def generate_cubic(self, frames):  
        return generate_cubic(frames)

    def get_split_frames(self, max_to_extract=2000):  # * if you'd like to only work on a sample, slice this list
        # Backwards-compatible helper that returns up to `max_to_extract` unique frames
        # starting at the beginning of the video.
        self.frames = split_frames(self.video_path, max_to_extract)
        return self.frames  # [100:200]

    def get_frames_window(self, start_frame: int = 0, max_to_extract: int = 2000, threshold: float = 2.0):
        # New helper: returns (frames_list, last_raw_index) for a window starting
        # at `start_frame`. Uses split_frames_window implemented in preprocessing.
        from src.tasks.preprocessing import split_frames_window
        frames, last_idx = split_frames_window(self.video_path, start_frame=start_frame, threshold=threshold, max_to_extract=max_to_extract)
        return frames, last_idx


class VideoProcessor:
    def __init__(self, video_loader: VideoLoader, cubic):
        self.video_loader = video_loader
        self.cubic = cubic
        if cubic:
            # keep this for backward compatibility; code that provides explicit
            # frames will pass them into methods instead
            self.cubic_frames = None

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

    def get_depth_mask_for(self, frames):
        # New API: compute depths for either cubic dict or list of frames
        if self.cubic:
            return predict_cubic_depths(frames)
        return predict_depths(frames)

    def segment(self, object_name=None):
        # for this model there is no object name so we ignore for now
        if self.cubic:
            frames = self.cubic_frames
            segmentation_masks = predict_cubic_segmentations(frames)
            return segmentation_masks
        frames = self.video_loader.frames
        segmentation_masks = predict_segmentations(frames)
        return segmentation_masks

    def segment_for(self, frames, object_name=None):
        # New API: segmentation for provided frames (either cubic dict or list)
        if self.cubic:
            return predict_cubic_segmentations(frames)
        return predict_segmentations(frames)

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
    def __init__(self, video_path, cubic=True):
        self.video_loader = VideoLoader(video_path)
        self.video_processor = VideoProcessor(self.video_loader, cubic=cubic)

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
        for i, frame_segments in enumerate(segmented_items):
            if not frame_segments:
                continue

            # Sort closest to furthest
            sorted_segments = sorted(frame_segments, key=lambda x: x["mode_depth"])

            # Keep elements that fall within valid depth boundary [0.1, depth_threshold)
            segmented_items[i] = [
                item for item in sorted_segments 
                if item["mode_depth"] is not None and 0.1 <= item["mode_depth"] < depth_threshold
            ]

        return segmented_items

    def process_vision(self, object_name=None):
        # Backwards-compatible default behaviour (no frames provided): run as before
        depth_masks = self.video_processor.get_depth_mask()
        segmentation_masks = self.video_processor.segment(object_name)
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
        segmented_items = self.prune_depth(segmented_items)         
        environment_items = self.prune_depth(environment_items)
      
        return segmented_items, environment_items

    # * is the same as the function i wrote above, but just takes specific frames as input, above function will be removed later
    def process_vision_for(self, frames, object_name=None):
        """
        Process a provided list of frames (or cubic dict) and return segmented items.
        This allows sliding-window processing without changing the core logic.
        """
        # compute depth masks and segmentation masks for provided frames
        if self.video_processor.cubic:
            cubic_frames = self.video_loader.generate_cubic(frames)
            depth_masks = self.video_processor.get_depth_mask_for(cubic_frames)
            segmentation_masks = self.video_processor.segment_for(cubic_frames, object_name)
            frames_front = cubic_frames["front"]
        else:
            depth_masks = self.video_processor.get_depth_mask_for(frames)
            segmentation_masks = self.video_processor.segment_for(frames, object_name)
            frames_front = frames

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
        segmented_items = self.prune_depth(segmented_items)         
        environment_items = self.prune_depth(environment_items)
        
        return segmented_items, environment_items

    # * here is where the environment description is generated, the detected static elements will be added here next time
    def process_environment(self, static_objects, prompt: str = ENV_PROMPT) -> str:
        # ! add the input of the static objects segmentation
        """
        Produces a structured environment description for the current video clip
        by querying the world model (environment.py) on the front-facing frames.

        The front camera is used because it provides the rider's primary field of
        view and is the most relevant for detecting road conditions, traffic signs,
        and weather. The description is a single string (structured JSON from the
        VLM) that is shared across all frames when calling process_risk().

        Args:
            prompt: VLM prompt to use. Defaults to ENV_PROMPT from the config.

        Returns:
            Environment description string (structured JSON from the world model).
        """
        # Backwards-compatible behaviour: use preloaded cubic frames if present
        if self.video_processor.cubic and self.video_processor.cubic_frames is not None:
            front_frames = self.video_processor.cubic_frames["front"]
        elif self.video_processor.cubic and self.video_loader.frames is not None:
            front_frames = self.video_loader.generate_cubic(self.video_loader.frames)["front"]
        elif not self.video_processor.cubic and self.video_loader.frames is not None:
            front_frames = self.video_loader.frames
        else:
            # No preloaded frames — fall back to loading up to 2000 frames from start
            front_frames = self.video_loader.get_split_frames()

        # add static objects to prompt
        resolved_prompt = prompt.format(
            STATIC_OBJECTS=str(static_objects) if static_objects is not None else "[]"
        )

        environment_descriptions = []
        # run environment by each 10 frames
        for i in range(0, len(front_frames), 10):  # ! implement tqdm later
            env_description = query_world_model(
                prompt=resolved_prompt,
                images=front_frames[i:i+10],
                model=CONFIG["vlm"]["world_model"]["model_name"],
            )
            environment_descriptions.append(env_description)

        # normalise to a single string in both cases.
        if isinstance(environment_descriptions[-1], list):
            env_description = "\n".join(environment_descriptions[-1])
        else:
            env_description = environment_descriptions[-1]

        return env_description

    # * the full risk processing pipeline, using sliding window of 50 frames (assuming framerate of gopro is 50)
    def process_risk(
            self,
            segmented_items: list[list[dict]],
            env_description: str,
            telemetry: Optional[TelemetryData] = None,
            api_base: str = CONFIG["risk_assessment"]["api_base"],
            model_name: str = CONFIG["risk_assessment"]["model_name"],
            epoch_window_size: Optional[int] = CONFIG["risk_assessment"].get("epoch_window_size", 50),
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

            if epoch_window_size and epoch_window_size < len(segmented_items):
                epoch_results = []
                for i in range(0, len(segmented_items), epoch_window_size):
                    window = segmented_items[i : i + epoch_window_size]
                    res = engine.assess_epoch(
                        segmented_items=window,
                        env_description=env_description,
                        telemetry=telemetry,
                    )
                    epoch_results.append(res)
                return epoch_results
            else:
                return engine.assess_epoch(
                    segmented_items=segmented_items,
                    env_description=env_description,
                    telemetry=telemetry,
                )
