# for parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from src.tasks.config.utils import CONFIG

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

    def process(self, object_name=None):
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