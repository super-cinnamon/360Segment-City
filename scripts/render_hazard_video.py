import os
import sys
import json
import pickle
import cv2
import numpy as np
import argparse
from pathlib import Path

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.telemetry import TelemetryProcessor
from src.tasks.risk_assessment import (
    TelemetryData,
    DetectedAgent,
    build_agents_from_segments,
    compute_depth_bounds,
    generate_hazard_viz_video
)
from src.tasks.config.utils import CONFIG

def load_cache(cache_dir, prefix, video_name, idx_str):
    path = Path(cache_dir) / f"{prefix}_{video_name}_{idx_str}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def extract_frames(video_path, start_frame, count):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB to match pipeline
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def main():
    parser = argparse.ArgumentParser(description="Standalone Hazard Visualization Renderer")
    parser.add_argument("--video", type=str, required=True, help="Path to source 360 video")
    parser.add_argument("--risk_data", type=str, required=True, help="Path to detailed risk JSONL")
    parser.add_argument("--telemetry_dir", type=str, required=True, help="Directory containing telemetry CSVs")
    parser.add_argument("--output", type=str, default="outputs/hazard_viz_standalone.mp4", help="Output video path")
    parser.add_argument("--cache_dir", type=str, default="data/cache/vision", help="Vision cache directory")
    args = parser.parse_args()

    # 1. Filter flagged epochs
    flagged_epochs = []
    with open(args.risk_data, "r") as f:
        for line in f:
            data = json.loads(line)
            meta = data.get("meta", {})
            res = data.get("result", {})
            if res and res.get("reasons"):
                # Only include if there are hazards
                has_hazards = any(r.get("hazards") for r in res["reasons"])
                if has_hazards:
                    flagged_epochs.append({
                        "idx": meta.get("epoch_idx"),
                        "start": meta.get("raw_start"),
                        "end": meta.get("raw_end"),
                        "reasons": res["reasons"]
                    })

    if not flagged_epochs:
        print("No flagged epochs found in risk data.")
        return

    print(f"Found {len(flagged_epochs)} flagged epochs. Starting render...")

    # 2. Load Telemetry
    tp = TelemetryProcessor(args.telemetry_dir)

    video_name = Path(args.video).name
    viz_videos = []

    # 3. Process each flagged epoch
    for ep in flagged_epochs:
        idx = ep["idx"]
        start = ep["start"]
        end = ep["end"]

        print(f"Rendering Epoch {idx} (frames {start}-{end})...")

        # Reconstruct frames (Composite 360 View simulation)
        # In a real scenario, we'd need the 4 views.
        # Since we are simplified, we'll just extract the front view or assume
        # the user provides the composite video.
        # But the hazard_viz_video takes a dict of sides.
        # To be fully compatible, we'd need 4 videos or a way to split the composite.
        # For this standalone script, we will assume the input --video is the source 360 video
        # and we only simulate the 'front' side for simplicity unless otherwise specified.

        frames_list = extract_frames(args.video, start, end - start)
        frames = {"front": frames_list}

        # Retrieve Caches
        idx_str = f"{start}_{end}"
        seg_cache = load_cache(args.cache_dir, "seg", video_name, idx_str)
        depth_cache = load_cache(args.cache_dir, "depth", video_name, idx_str)

        if seg_cache is None or depth_cache is None:
            print(f"Warning: Missing cache for epoch {idx}. Skipping.")
            continue

        # Reconstruct DetectedAgents
        # seg_cache is a list of lists (per frame) of segmentation labels
        # we need to convert this to segmented_items format
        # [ [ {"mask":..., "class_name":...}, ... ], ... ]

        # This part is tricky because the cache stores the raw model output.
        # We need to emulate the process_vision logic to get a list of agents.

        # For brevity in this script, we'll use a simplified agent reconstruction
        # that uses the cached segmentation maps.

        # This is where it gets complex. The cache stores the 'segmentation_masks'
        # which are list of { "segmentation_map": ..., "segmentation_labels": ... }

        epoch_agents = []
        for f_idx in range(len(frames_list)):
            frame_segs = []
            # Only handle 'front' for this standalone renderer simplification
            if "front" not in seg_cache or f_idx >= len(seg_cache["front"]):
                continue
            seg_info = seg_cache["front"][f_idx]
            for label in seg_info["segmentation_labels"]:
                mask = (seg_info["segmentation_map"] == label["id"])
                class_name = CONFIG["segmentation"]["id2label"].get(str(label["label_id"]), "Unknown")
                frame_segs.append({
                    "mask": mask,
                    "class_name": class_name,
                    "score": label.get("score", 1.0),
                    "side": "front",
                    "mode_depth": np.mean(depth_cache["front"][1][f_idx][mask]) if (np.any(mask) and "front" in depth_cache and len(depth_cache["front"]) > 1 and len(depth_cache["front"][1]) > f_idx) else 1.0
                })

            # Now use the actual helper from risk_assessment
            d_min, d_max = compute_depth_bounds([frame_segs]) # simplified
            agents = build_agents_from_segments(frame_segs, d_min, d_max)
            epoch_agents.append(agents)

        # Telemetry for this epoch
        epoch_telemetry = []
        if tp and tp.data is not None:
            df = tp.data
            start_t = idx * 30
            end_t = start_t + 30
            if start_t < len(df):
                epoch_df = df.iloc[start_t:end_t]
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

        # Render
        out_path = f"temp_viz_{idx}.mp4"
        generate_hazard_viz_video(
            output_path=out_path,
            frames=frames,
            epoch_agents=epoch_agents,
            reasons_detail=ep["reasons"],
            epoch_telemetry=epoch_telemetry
        )
        viz_videos.append(out_path)

    # Merge videos
    if viz_videos:
        print(f"Merging {len(viz_videos)} videos into {args.output}...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cap = cv2.VideoCapture(viz_videos[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        for v in viz_videos:
            cap = cv2.VideoCapture(v)
            while True:
                ret, frame = cap.read()
                if not ret: break
                out.write(frame)
            cap.release()
            os.remove(v)
        out.release()
        print("Done.")
    else:
        print("No videos were generated.")

if __name__ == "__main__":
    main()
