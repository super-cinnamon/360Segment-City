"""
Hazard Detection Visualization Module
=====================================

Handles logging and visualization for the two-stage hazard detection pipeline:
- VLM inputs/outputs logging
- YOLOE detection results
- Merged agents with hazard masks
- Video generation for hazard detection visualization
"""

import os
import json
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.tasks.config.utils import CONFIG

logger = logging.getLogger(__name__)

# Output directory for hazard detection logs and visualizations
HAZARD_VIZ_DIR = Path("outputs/hazard_detection")
HAZARD_VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Subdirectories
VLM_LOGS_DIR = HAZARD_VIZ_DIR / "vlm_logs"
YOLOE_LOGS_DIR = HAZARD_VIZ_DIR / "yoloe_logs"
MERGED_LOGS_DIR = HAZARD_VIZ_DIR / "merged_agents"
FRAMES_DIR = HAZARD_VIZ_DIR / "frames"
VIDEOS_DIR = HAZARD_VIZ_DIR / "videos"

for d in [VLM_LOGS_DIR, YOLOE_LOGS_DIR, MERGED_LOGS_DIR, FRAMES_DIR, VIDEOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class HazardDetectionLogger:
    """
    Logs all inputs and outputs of the hazard detection pipeline
    for debugging and visualization.
    """

    def __init__(self, epoch_idx: int = 0, run_id: str = None):
        self.epoch_idx = epoch_idx
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = HAZARD_VIZ_DIR / f"epoch_{epoch_idx}_{self.run_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for this session
        self.vlm_dir = self.session_dir / "vlm"
        self.yoloe_dir = self.session_dir / "yoloe"
        self.merged_dir = self.session_dir / "merged"
        self.frames_dir = self.session_dir / "frames"
        for d in [self.vlm_dir, self.yoloe_dir, self.merged_dir, self.frames_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.frame_counter = 0
        self.video_writer = None
        self.video_path = None

        logger.info(f"HazardDetectionLogger initialized: {self.session_dir}")

    def log_vlm_input(self, image: np.ndarray, prompt: str, speed_kmh: float, lean_angle_deg: float):
        """Log VLM input: image, prompt, and motorcycle state."""
        # Save input image
        img_path = self.vlm_dir / f"input_frame_{self.frame_counter:04d}.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Save prompt and state
        prompt_path = self.vlm_dir / f"prompt_{self.frame_counter:04d}.txt"
        with open(prompt_path, "w") as f:
            f.write(f"=== MOTORCYCLE STATE ===\n")
            f.write(f"Speed: {speed_kmh} km/h\n")
            f.write(f"Lean Angle: {lean_angle_deg} degrees\n")
            f.write(f"\n=== PROMPT ===\n")
            f.write(prompt)

        logger.debug(f"Logged VLM input: {img_path}, {prompt_path}")

    def log_vlm_output(self, response: str, parsed_hazards: List[Dict]):
        """Log VLM output: raw response and parsed hazards."""
        # Save raw response
        response_path = self.vlm_dir / f"response_{self.frame_counter:04d}.txt"
        with open(response_path, "w") as f:
            f.write(response)

        # Save parsed hazards as JSON
        hazards_path = self.vlm_dir / f"hazards_{self.frame_counter:04d}.json"
        with open(hazards_path, "w") as f:
            json.dump({
                "frame": self.frame_counter,
                "timestamp": datetime.now().isoformat(),
                "num_hazards": len(parsed_hazards),
                "hazards": parsed_hazards
            }, f, indent=2)

        # Log summary
        logger.info(f"VLM identified {len(parsed_hazards)} hazards:")
        for h in parsed_hazards:
            logger.info(f"  - {h['label']}: c_vlm={h.get('c_vlm', 0):.2f}, type={h.get('hazard_type', 'unknown')}")

    def log_yoloe_results(self, image: np.ndarray, hazard_labels: List[str], yoloe_results: List[Dict]):
        """Log YOLOE segmentation results with visualizations."""
        # Save YOLOE input image
        img_path = self.yoloe_dir / f"input_frame_{self.frame_counter:04d}.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Save YOLOE results
        results_path = self.yoloe_dir / f"results_{self.frame_counter:04d}.json"
        serializable_results = []
        for r in yoloe_results:
            sr = {
                "label": r["label"],
                "confidence": float(r["confidence"]),
                "bbox": [int(x) for x in r["bbox"]],
                "class_id": int(r["class_id"]),
                "c_vlm": float(r.get("c_vlm", 0)),
                "hazard_type": r.get("hazard_type", "unknown"),
                "description": r.get("description", ""),
                "mask_shape": r["mask"].shape,
                "mask_area": int(r["mask"].sum())
            }
            serializable_results.append(sr)

        with open(results_path, "w") as f:
            json.dump({
                "frame": self.frame_counter,
                "hazard_labels": hazard_labels,
                "num_detections": len(yoloe_results),
                "detections": serializable_results
            }, f, indent=2)

        # Create visualization: overlay masks on image
        viz_image = self._create_yoloe_viz(image, yoloe_results)
        viz_path = self.yoloe_dir / f"viz_{self.frame_counter:04d}.jpg"
        cv2.imwrite(str(viz_path), cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))

        # Also save individual masks
        for i, r in enumerate(yoloe_results):
            mask_path = self.yoloe_dir / f"mask_{self.frame_counter:04d}_{r['label']}.png"
            cv2.imwrite(str(mask_path), (r["mask"].astype(np.uint8) * 255))

        logger.info(f"YOLOE detected {len(yoloe_results)} hazards:")
        for r in yoloe_results:
            logger.info(f"  - {r['label']}: conf={r['confidence']:.2f}, c_vlm={r.get('c_vlm', 0):.2f}, area={r['mask'].sum()}")

    def log_merged_agents(self, image: np.ndarray, depth_map: np.ndarray,
                          segmented_agents: List, yoloe_results: List[Dict],
                          merged_agents: List, merger_info: Dict = None):
        """Log merged agents with visualizations."""
        # Save merged agents summary
        agents_path = self.merged_dir / f"agents_{self.frame_counter:04d}.json"

        agent_summaries = []
        for a in merged_agents:
            summary = {
                "agent_id": a.agent_id,
                "class_name": a.class_name,
                "seg_score": float(a.seg_score),
                "mode_depth": float(a.mode_depth) if a.mode_depth else None,
                "distance_weight": float(a.distance_weight),
                "side": a.side,
                "fused_sides": a.fused_sides,
                "was_fused": a.was_fused,
                "hazard_verified": getattr(a, 'hazard_verified', False),
                "is_environment_hazard": getattr(a, 'is_environment_hazard', False),
                "mask_area": int(a.masks.get("front", np.zeros((1,1))).sum()) if "front" in a.masks else 0
            }
            if hasattr(a, 'hazard_metadata'):
                summary["hazard_metadata"] = a.hazard_metadata
            agent_summaries.append(summary)

        with open(agents_path, "w") as f:
            json.dump({
                "frame": self.frame_counter,
                "num_segmented_agents": len(segmented_agents),
                "num_yoloe_hazards": len(yoloe_results),
                "num_merged_agents": len(merged_agents),
                "agents": agent_summaries,
                "merger_info": merger_info
            }, f, indent=2)

        # Create visualization: overlay all merged agent masks
        viz_image = self.create_merged_viz(image, depth_map, merged_agents)
        viz_path = self.merged_dir / f"viz_{self.frame_counter:04d}.jpg"
        cv2.imwrite(str(viz_path), cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))

        # Also save depth map visualization
        depth_viz = self._create_depth_viz(depth_map)
        depth_path = self.merged_dir / f"depth_{self.frame_counter:04d}.jpg"
        cv2.imwrite(str(depth_path), depth_viz)

        logger.info(f"Merged agents: {len(merged_agents)} total")
        for a in merged_agents:
            hazard_str = ""
            if hasattr(a, 'hazard_metadata') and a.hazard_metadata:
                hazard_str = f" | hazards: {', '.join([h['label'] for h in a.hazard_metadata])}"
            env_str = " [ENV HAZARD]" if getattr(a, 'is_environment_hazard', False) else ""
            logger.info(f"  - {a.agent_id}: {a.class_name}, score={a.seg_score:.2f}, depth={a.mode_depth:.2f}{hazard_str}{env_str}")

    def _create_yoloe_viz(self, image: np.ndarray, yoloe_results: List[Dict]) -> np.ndarray:
        """Create visualization of YOLOE detections."""
        viz = image.copy()

        # Color palette for different hazard types
        colors = {
            "environment_hazard": (255, 165, 0),    # Orange
            "road_condition": (255, 0, 255),        # Magenta
            "dynamic_agent": (0, 255, 0),           # Green
            "unknown": (128, 128, 128)              # Gray
        }

        for r in yoloe_results:
            mask = r["mask"]
            label = r["label"]
            conf = r["confidence"]
            hazard_type = r.get("hazard_type", "unknown")
            c_vlm = r.get("c_vlm", 0)

            color = colors.get(hazard_type, colors["unknown"])

            # Apply mask overlay
            overlay = viz.copy()
            overlay[mask] = color
            cv2.addWeighted(overlay, 0.4, viz, 0.6, 0, viz)

            # Draw bounding box
            bbox = r["bbox"]
            cv2.rectangle(viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Draw label
            label_text = f"{label}: conf={conf:.2f}, c_vlm={c_vlm:.2f}"
            cv2.putText(viz, label_text, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return viz

    def create_merged_viz(self, image: np.ndarray, depth_map: np.ndarray,
                           merged_agents: List) -> np.ndarray:
        """Create visualization of merged agents."""
        viz = image.copy()
        h, w = image.shape[:2]

        # Color palette
        base_colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]

        for i, a in enumerate(merged_agents):
            mask = a.masks.get("front")
            if mask is None or not np.any(mask):
                continue

            color = base_colors[i % len(base_colors)]

            # Different overlay style for environment hazards
            if getattr(a, 'is_environment_hazard', False):
                # Solid fill for environment hazards
                overlay = viz.copy()
                overlay[mask] = color
                cv2.addWeighted(overlay, 0.5, viz, 0.5, 0, viz)
            else:
                # Semi-transparent for regular agents
                overlay = viz.copy()
                overlay[mask] = color
                cv2.addWeighted(overlay, 0.3, viz, 0.7, 0, viz)

            # Draw bounding box
            coords = np.argwhere(mask)
            if len(coords) > 0:
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                cv2.rectangle(viz, (x0, y0), (x1, y1), color, 2)

                # Draw agent info
                label_parts = [a.agent_id, a.class_name, f"s={a.seg_score:.2f}"]
                if hasattr(a, 'hazard_metadata') and a.hazard_metadata:
                    for hm in a.hazard_metadata:
                        label_parts.append(f"[{hm['label']}:v={hm.get('c_vlm', 0):.2f}]")
                label = " | ".join(label_parts)

                cv2.putText(viz, label, (x0, max(y0 - 5, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Add legend
        self._add_legend(viz, merged_agents, base_colors)

        return viz

    def _create_depth_viz(self, depth_map: np.ndarray) -> np.ndarray:
        """Create depth map visualization."""
        # Normalize depth to 0-255
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        return depth_colored

    def _add_legend(self, image: np.ndarray, agents: List, colors: List):
        """Add legend to visualization."""
        legend_y = 30
        for i, a in enumerate(agents):
            if i >= len(colors):
                break
            color = colors[i]
            label = f"{a.agent_id} ({a.class_name})"
            if getattr(a, 'is_environment_hazard', False):
                label += " [ENV]"
            cv2.putText(image, label, (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            legend_y += 20

    def add_frame_to_video(self, frame: np.ndarray):
        """Add frame to video writer for continuous visualization."""
        if self.video_writer is None:
            h, w = frame.shape[:2]
            # Ensure even dimensions
            w = w - w % 2
            h = h - h % 2
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))

            self.video_path = VIDEOS_DIR / f"hazard_detection_epoch_{self.epoch_idx}_{self.run_id}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(self.video_path), fourcc, 10.0, (w, h))

        if self.video_writer is not None and self.video_writer.isOpened():
            # Convert RGB to BGR for video
            if frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            self.video_writer.write(frame_bgr)

    def finalize_video(self):
        """Close video writer and return path."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            logger.info(f"Hazard detection video saved: {self.video_path}")
            return str(self.video_path)
        return None

    def increment_frame(self):
        """Increment frame counter."""
        self.frame_counter += 1

    def get_session_dir(self) -> Path:
        """Return session directory path."""
        return self.session_dir


def create_hazard_logger(epoch_idx: int = 0, run_id: str = None) -> HazardDetectionLogger:
    """Factory function to create hazard detection logger."""
    return HazardDetectionLogger(epoch_idx=epoch_idx, run_id=run_id)


def visualize_risk_map(risk_map: np.ndarray, image: np.ndarray,
                       output_path: str, alpha: float = 0.5):
    """Create and save risk map visualization overlay."""
    # Normalize risk map to 0-255
    risk_norm = cv2.normalize(risk_map, None, 0, 255, cv2.NORM_MINMAX)
    risk_uint8 = risk_norm.astype(np.uint8)

    # Apply heatmap colormap
    risk_heatmap = cv2.applyColorMap(risk_uint8, cv2.COLORMAP_JET)

    # Overlay on original image
    if image.shape[:2] != risk_heatmap.shape[:2]:
        risk_heatmap = cv2.resize(risk_heatmap, (image.shape[1], image.shape[0]))

    overlay = cv2.addWeighted(image, 1 - alpha, risk_heatmap, alpha, 0)

    # Add colorbar
    h, w = overlay.shape[:2]
    bar_width = 30
    bar_height = h - 40
    bar_x = w - bar_width - 10
    bar_y = 20

    # Draw colorbar
    for i in range(bar_height):
        val = i / bar_height
        color = cv2.applyColorMap(np.array([[[int(val * 255)]]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0]
        cv2.line(overlay, (bar_x, bar_y + i), (bar_x + bar_width, bar_y + i),
                color.tolist(), 1)

    # Add labels
    cv2.putText(overlay, "High", (bar_x + bar_width + 5, bar_y + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(overlay, "Low", (bar_x + bar_width + 5, bar_y + bar_height - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(overlay, "Risk", (bar_x + bar_width + 5, bar_y + bar_height // 2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imwrite(output_path, overlay)
    return overlay