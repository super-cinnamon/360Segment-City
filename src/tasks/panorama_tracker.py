import os
import sys
import numpy as np
import torch
import cv2

# Path Configuration
base_dir = os.getcwd()
external_dir = os.path.join(base_dir, 'external_tracking')
sys.path.insert(0, external_dir)
sys.path.insert(0, os.path.join(external_dir, 'lib'))
sys.path.insert(0, os.path.join(base_dir, 'mock_detectron2'))

# Now import the actual OD logic
from panoramic_detection import improved_OD as OD
from strongsort.strong_sort_new import StrongSort

class PanoramaTracker:
    def __init__(
        self, 
        model_type="YOLO", 
        min_size=640, 
        score_threshold=0.4, 
        nms_threshold=0.45, 
        fov=120, 
        thetas=[0, 90, 180, 270], 
        phis=[-10, -10, -10, -10],
        front_yaw_offset=0.0,
        max_track_age=30,
        history_prune_limit=100
    ):
        self.fov = fov
        self.thetas = thetas
        self.phis = phis
        self.front_yaw_offset = front_yaw_offset
        self.history_prune_limit = history_prune_limit
        
        self.model, self.cfg, self.yolo_cfg = OD.load_model(
            model_type, min_size, 10000, 2.0, score_threshold, nms_threshold
        )
        
        ckpt_path = os.path.join(external_dir, 'deep_sort/deep/checkpoint/ckpt.t7')
        if not os.path.exists(ckpt_path):
            print(f"WARNING: StrongSORT checkpoint not found at {ckpt_path}.")
            
        self.tracker = StrongSort(
            ckpt_path, 
            use_cuda=torch.cuda.is_available()
        )
        
        self.frame_idx = 0
        self.history = {}

    def side_from_x(self, cx, width):
        yaw = (cx / width) * 360.0
        yaw = (yaw - self.front_yaw_offset) % 360
        return ["front", "right", "back", "left"][int(((yaw + 45) % 360) // 90)]

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        self.frame_idx += 1

        bboxes_all, classes_all, scores_all = OD.predict_one_frame(
            self.fov, self.thetas, self.phis, frame, self.model, w, h,
            [0, 1, 2, 3, 5, 7, 9], True, True, "YOLO", False, self.yolo_cfg
        )

        bboxes_xcycwh = OD.xyxy2xcycwh(bboxes_all)

        track_outputs = self.tracker.update(
            np.array(bboxes_xcycwh),
            np.array(classes_all),
            np.array(scores_all),
            frame,
            prevent_different_classes_match=True,
            match_across_boundary=True
        )

        current_tracks = []
        if len(track_outputs) > 0:
            for t in track_outputs:
                x1, y1, x2, y2, cls, score, tid = t
                cx = (x1 + x2) / 2
                side = self.side_from_x(cx, w)
                
                track_data = {
                    'id': int(tid),
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'class': int(cls),
                    'score': float(score),
                    'side': side,
                    'cx': cx
                }
                current_tracks.append(track_data)
                
                if tid not in self.history:
                    self.history[tid] = []
                self.history[tid].append((self.frame_idx, track_data['box'], side))

        self._prune_history()
        return current_tracks

    def _prune_history(self):
        for tid in list(self.history.keys()):
            if len(self.history[tid]) > self.history_prune_limit:
                self.history[tid] = self.history[tid][-self.history_prune_limit:]

    def get_track_history(self, track_id):
        return self.history.get(track_id, [])
