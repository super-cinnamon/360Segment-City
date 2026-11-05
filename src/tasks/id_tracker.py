import numpy as np

class ObjectIDTracker:
    """
    Tracks object IDs across frames within an epoch using mask IoU and labels.
    """
    def __init__(self, iou_threshold=0.5, max_missing_frames=30):
        self.cache = {}  # id -> {mask, label, side, last_frame}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self._assigned_frame = None
        self._assigned_ids = set()

    def get_id(self, current_mask, label, frame_idx, side=None):
        if frame_idx != self._assigned_frame:
            self._assigned_frame = frame_idx
            self._assigned_ids.clear()

        best_id = None
        max_iou = 0

        # Clean up old entries from cache to prevent drift and memory growth
        self._cleanup_cache(frame_idx)

        for obj_id, data in self.cache.items():
            if obj_id in self._assigned_ids:
                continue
            if data['label'] == label and data.get('side') == side:
                iou = self.calculate_iou(current_mask, data['mask'])
                if iou > max_iou:
                    max_iou = iou
                    best_id = obj_id

        if best_id is not None and max_iou >= self.iou_threshold:
            # Update cache with new position/mask
            self.cache[best_id] = {
                "mask": current_mask,
                "label": label,
                "side": side,
                "last_frame": frame_idx,
            }
            self._assigned_ids.add(best_id)
            return best_id
        else:
            # Create new ID
            new_id = self.next_id
            self.cache[new_id] = {
                "mask": current_mask,
                "label": label,
                "side": side,
                "last_frame": frame_idx,
            }
            self.next_id += 1
            self._assigned_ids.add(new_id)
            return new_id

    def _cleanup_cache(self, current_frame):
        """Remove objects not seen for max_missing_frames."""
        to_remove = [
            obj_id for obj_id, data in self.cache.items()
            if current_frame - data['last_frame'] > self.max_missing_frames
        ]
        for obj_id in to_remove:
            del self.cache[obj_id]

    def calculate_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0
