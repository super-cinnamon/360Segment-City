# for parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.tasks.preprocessing import (
    load_cubic,
    split_frames,
)
from src.tasks.depth_estimation import (
    predict_depths
)
from src.tasks.segmentation import predict_segmentation

class VideoLoader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.frames = split_frames(video_path)

    def load_video(self):
        return load_video(self.video_path)

    def load_cubic(self):
        return load_cubic(self.video_path)

    def split_frames(self):
        return split_frames(self.video_path)


class VideoProcessor:
    def __init__(self, video_loader: VideoLoader):
        self.video_loader = video_loader

    def get_depth_mask(self):
        pass

    def segment(self, object_name=None):
        # for this model there is no object name so we ignore for now
        pass

    def clean_segmentation(self, mask):
        pass

    