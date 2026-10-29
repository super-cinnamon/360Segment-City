import sys
import os
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path
import uuid

# Initialize logger for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add necessary paths for the Panorama Tracker and its dependencies
base_dir = os.getcwd()
external_dir = os.path.join(base_dir, 'external_tracking')
sys.path.append(base_dir)
sys.path.append(external_dir)
sys.path.append(os.path.join(external_dir, 'detectron2'))
sys.path.append(os.path.join(external_dir, 'lib'))

# Check for the mandatory StrongSORT checkpoint
ckpt_path = os.path.join(external_dir, 'deep_sort/deep/checkpoint/ckpt.t7')
if not os.path.exists(ckpt_path):
    print(f"CRITICAL ERROR: StrongSORT checkpoint not found at {ckpt_path}")
    print("Please download it using: gdown 'https://drive.google.com/uc/export/download&id=1_qwTWdzT9dWNudpusgKavj_4elGgbkUN' -O " + ckpt_path)
    # We don't sys.exit(1) here to allow the pipeline to attempt to run,
    # but tracking will be degraded.

# for parallel processing
import json
import gc
import cv2

from src.process import SegmentationPipeline
from src.tasks.api.api_run import start_api_in_background
from src.tasks.environment import query_world_model
from src.tasks.config.utils import ENV_PROMPT, CONFIG, apply_vlm_profile

DATA_PATH = "data/GS020012_last30s.mp4"
OUTPUT_DIR = "data/outputs"


def ensure_out_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def merge_hazard_videos(video_paths, output_path):
    if not video_paths:
        print("No videos to merge.")
        return

    print(f"Merging {len(video_paths)} hazard videos into {output_path}...", flush=True)

    # Get properties from the first video
    cap = cv2.VideoCapture(video_paths[0])
    if not cap.isOpened():
        print(f"Error opening first video: {video_paths[0]}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {path}. Skipping.")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print(f"Successfully merged hazard video: {output_path}", flush=True)

def run_full_pipeline(sample_dir="data/sample_1", model_profile=None):
    """
    Runs the full risk assessment pipeline for a given directory.
    Expects '360_video.mp4' and telemetry CSVs in the directory.
    """
    sample_path = Path(sample_dir).resolve()
    sample_name = sample_path.name
    profile = apply_vlm_profile(model_profile)
    run_id = uuid.uuid4().hex
    run_name = f"{profile['id']}_{run_id}"
    run_output_dir = Path(OUTPUT_DIR) / sample_name / run_name
    viz_output_dir = Path("outputs/risk_viz") / sample_name / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    viz_output_dir.mkdir(parents=True, exist_ok=True)
    detailed_out = run_output_dir / "paper_risk_epochs_detailed.jsonl"
    summary_out = run_output_dir / "paper_risk_epochs_summary.jsonl"
    source_video = sample_path / "360_video.mp4"
    run_metadata = {
        "run_id": run_id,
        "model_profile": profile["id"],
        "model_label": profile["label"],
        "model_repo_id": profile["repo_id"],
        "model_backend": profile["backend"],
        "sample_dir": str(sample_path),
        "started_at": datetime.now().isoformat(),
    }
    with open(run_output_dir / "run_metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(run_metadata, metadata_file, indent=2)

    print(f"Starting full pipeline for directory: {sample_path}", flush=True)

    # Initialize pipeline with directory path. This automatically triggers TelemetryProcessor
    # initialization if the directory contains valid GPMF CSVs.
    pipeline = SegmentationPipeline(str(sample_path), cubic=True)

    epoch_size = CONFIG["risk_assessment"].get("epoch_window_size", 60)
    chunk_max_frames = epoch_size * 5

    start_raw = 0
    epoch_idx = 0
    viz_videos = []
    force_next_epoch = False

    # Setup output directory
    ensure_out_dir()

    while True:
        print(f"DEBUG: Getting frames window starting at {start_raw}...", flush=True)
        frames, last_raw = pipeline.video_loader.get_frames_window(start_raw, max_to_extract=chunk_max_frames)
        if not frames:
            logger.info("No more frames found. Ending pipeline processing.")
            break

        for i in range(0, len(frames), epoch_size):
            epoch_frames = frames[i : i + epoch_size]
            actual_start = start_raw + i
            actual_end = start_raw + min(i + epoch_size, len(frames))
            force_current_epoch = force_next_epoch
            force_next_epoch = False

            result = pipeline.process_epoch(
                epoch_frames,
                epoch_idx=epoch_idx,
                raw_start=actual_start,
                raw_end=actual_end,
                force_process=force_current_epoch,
                viz_output_dir=str(viz_output_dir.resolve()),
            )
            force_next_epoch = bool(
                result["processed"] and result.get("roi_triggered", False)
            )

            meta = {
                "epoch_idx": epoch_idx,
                "raw_start": actual_start,
                "raw_end": actual_end,
                "run_id": run_id,
                "model_profile": profile["id"],
                "model_repo_id": profile["repo_id"],
            }

            if result["processed"]:
                res = result["risk_result"]
                save_jsonl(detailed_out, {"meta": meta, "result": res})
                save_jsonl(summary_out, {"meta": meta, "final_score": res.get("final_score", res.get("expected_risk_score", None))})
                if "viz_video_path" in res:
                    viz_videos.append(res["viz_video_path"])
            else:
                # Log why the epoch was skipped (e.g., ROI gating)
                logger.info(f"Epoch {epoch_idx} skipped: {result['reason']}")
                save_jsonl(detailed_out, {"meta": meta, "result": {"processed": False, "reason": result["reason"]}})

            pipeline.cleanup_gpu()
            epoch_idx += 1

        del frames
        gc.collect()
        if last_raw is None: break
        start_raw = last_raw + 1

    merge_hazard_videos(viz_videos, str(viz_output_dir / "hazard_viz_full.mp4"))

    final_video = viz_output_dir / "hazard_analysis_full.mp4"
    render_command = [
        sys.executable,
        "scripts/render_hazard_video.py",
        "--video", str(source_video),
        "--risk-data", str(detailed_out),
        "--telemetry-dir", str(sample_path),
        "--sidecar-dir", str(viz_output_dir.resolve()),
        "--output", str(final_video),
        "--mask-threshold", "0",
    ]
    print("Generating final hazard analysis video...", flush=True)
    subprocess.run(render_command, check=True)
    print(
        f"Processing complete for {sample_path}. Results: {run_output_dir}; "
        f"final video: {final_video}",
        flush=True,
    )

def main():
    parser = argparse.ArgumentParser(description="Run the full panoptic perception and risk assessment pipeline.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to see fragment merging details.")
    parser.add_argument("--sample_dir", type=str, default="data/sample_1", help="Directory containing video and telemetry.")
    parser.add_argument("--model", type=str, help="VLM profile ID from src/tasks/config/config.json (overrides vlm.selected_model for this run).")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    ensure_out_dir()

    profile = apply_vlm_profile(args.model)
    print(f"Starting VLM profile: {profile['id']} ({profile['repo_id']})", flush=True)
    print("DEBUG: Starting API server...", flush=True)
    start_api_in_background(model_name=profile["repo_id"])

    run_full_pipeline(sample_dir=args.sample_dir, model_profile=profile["id"])

if __name__ == "__main__":
    main()
