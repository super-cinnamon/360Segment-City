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
        self.frames = split_frames(video_path)

    def load_cubic(self):  # ! to double check, need cubic generator function
        return load_cubic(self.video_path)
    
    def generate_cubic(self):
        return generate_cubic(self.frames)

    def get_split_frames(self):
        return self.frames


class VideoProcessor:
    def __init__(self, video_loader: VideoLoader, cubic):
        self.video_loader = video_loader
        self.cubic = cubic
        if cubic:
            self.cubic_frames = self.video_loader.generate_cubic()

    def get_depth_mask(self):
        if self.cubic:
            frames = self.cubic_frames[-5:]  # Get the last 5 frames, remove later
            depths = predict_cubic_segmentations(frames)
            return depths
        else:
            frames = self.video_loader.frames[-5:]  # Get the last 5 frames, remove later
            depths = predict_depths(frames)
            return depths

    def segment(self, object_name=None):
        # for this model there is no object name so we ignore for now
        if self.cubic:
            frames = self.cubic_frames
            frames["front"] = frames["front"][-5:]  # Get the last 5 frames, remove later
            frames["right"] = frames["right"][-5:]  # Get the last 5 frames, remove later
            frames["back"] = frames["back"][-5:]  # Get the last 5 frames, remove later
            frames["left"] = frames["left"][-5:]  # Get the last 5 frames, remove later

            segmentation_masks = predict_cubic_segmentations(frames)
            return segmentation_masks
        frames = self.video_loader.frames
        segmentation_masks = predict_segmentations(frames)
        return segmentation_masks

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

    def prune_segmentation(items, score_threshold=CONFIG["segmentation"]["score_threshold"], relevant_labels=CONFIG["segmentation"]["relevant_labels"]):
        """
        This method aims to prune out all segmentation masks that show a score below the given threshold.
        And also to prune out any label that is irrelevant to our needs. Both are given as input, and will 
        default to the config file defaults.
        This processes a single frame of segmentation items, and returns the pruned list of items.
        """

        for item in items:
            # Prune out items with a score below the threshold or irrelevant labels
            if item["score"] < score_threshold or item["class_id"] not in relevant_labels:
                items.remove(item)

        return items

    def prune_depth(segmented_items, depth_threshold=CONFIG["segmentation"]["depth_threshold"]):
        """
        This method aims to prune out items that are too far or too close to the camera.
        The closest item to the camera always gets pruned out if its score is below 0.1.
        The furthest items with a score of 0.9 or higher will be pruned out. values will be set in config.
        """
        for frame_segments in segmented_items:
            # Sort the segments by mode_depth
            sorted_segments = sorted(frame_segments, key=lambda x: x["mode_depth"])

            # Prune the closest item if its score is below 0.1
            if sorted_segments[0]["score"] < 0.1:
                frame_segments.remove(sorted_segments[0])

            # Prune out the furthest items with a score of 0.8 or higher
            for segment in sorted_segments[::-1]:  # Start from the furthest
                if segment["score"] >= depth_threshold:
                    frame_segments.remove(segment)
                else:
                    break  # Stop once we hit a segment that doesn't meet the criteria

        return segmented_items

    def process_vision(self, object_name=None):
        depth_masks = self.video_processor.get_depth_mask()
        segmentation_masks = self.video_processor.segment(object_name)
        # create the list of segmented items with their class names and which frame they belong to
        segmented_items = []
        for i in range (len(segmentation_masks["front"])): # looping through frames
            frame_segments = []
            for key in segmentation_masks.keys():  # looping through sides
                for segment_info in segmentation_masks[key][i]["segmentation_labels"]:  # loop through segmented items
                    # Retrieve human-readable class name from model's id2label mapping
                    class_name = CONFIG["segmentation"]["id2label"].get(str(segment_info["label_id"]), f"Class_{segment_info['label_id']}")
                    binary_mask = (segmentation_masks[key][i]["segmentation_map"] == segment_info["id"])
                    # calculate the mode of the depth for this object
                    mode_depth_value = mode_depth(depth_masks[key][i], binary_mask)
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
            frame_segments = self.prune_segmentation(frame_segments)
            segmented_items.append(frame_segments)
        # prune segmentation items based on depth
        segmented_items = self.prune_depth(segmented_items)         
      
        return segmented_items

    def process_environment(self, prompt: str = ENV_PROMPT) -> str:
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
        if self.video_processor.cubic:
            # Use front-facing cubic frames for the environment description
            front_frames = self.video_processor.cubic_frames["front"]
        else:
            front_frames = self.video_loader.frames

        env_description = query_world_model(
            prompt=prompt,
            images=front_frames,
            model=CONFIG["vlm"]["world_model"]["model_name"],
        )

        # query_world_model returns a string (Windows) or list of strings (Linux/vLLM);
        # normalise to a single string in both cases.
        if isinstance(env_description, list):
            env_description = "\n".join(env_description)

        return env_description

    def process_risk(
        self,
        segmented_items: list[list[dict]],
        env_description: str,
        telemetry: Optional[TelemetryData] = None,
        api_base: str = CONFIG["risk_assessment"]["api_base"],
        model_name: str = CONFIG["risk_assessment"]["model_name"],
    ) -> list[dict]:
        """
        Runs G-Eval risk assessment for every frame in segmented_items.

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

        Returns:
            List of G-Eval result dicts (one per frame), each containing:
                - "expected_risk_score" : float in [1, 3]
                - "score_probabilities" : {1: float, 2: float, 3: float}
                - "context_summary"     : metadata dict
        """
        engine = RiskAssessmentEngine(api_base=api_base, model_name=model_name)

        # Compute global depth bounds across all frames for consistent normalization
        depth_min, depth_max = compute_depth_bounds(segmented_items)

        risk_results = []
        for frame_segments in segmented_items:
            result = engine.assess_frame(
                frame_segments=frame_segments,
                env_description=env_description,
                telemetry=telemetry,
                depth_min=depth_min,
                depth_max=depth_max,
            )
            risk_results.append(result)

        return risk_results