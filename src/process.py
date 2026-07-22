# for parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.tasks.preprocessing import (
    load_cubic,
    split_frames,
    generate_cubic,
)
from src.tasks.depth_estimation import (
    predict_depths,
    get_closest_depth_mask
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

    def process(self, object_name=None):
        # depth_masks = self.video_processor.get_depth_mask()
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
                    frame_segments.append({
                        "frame": i,
                        "side": key,
                        "class_name": class_name,
                        "class_id": segment_info["label_id"],
                        "score": segment_info.get("score", None),
                        "was_fused": segment_info.get("was_fused", False),
                        "mask": binary_mask
                    })
            segmented_items.append(frame_segments)
                       
        # cleaned_segmentation_masks = self.video_processor.clean_segmentation(depth_masks, segmentation_masks)
        return segmented_items