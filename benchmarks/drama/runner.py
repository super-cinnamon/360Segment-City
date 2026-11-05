
import os
import json
import logging
import argparse
import cv2
import tempfile
import shutil
import copy
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from benchmarks.drama.dataset_loader import DramaLoader, DramaLayout
from src.process import SegmentationPipeline
from src.tasks.config.utils import CONFIG, apply_vlm_profile, resolve_vlm_profile

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_temp_video(frames_dir: str, clip_id: str) -> str:
    """
    Creates a temporary .mp4 video from PNG frames to feed into the pipeline.
    DRAMA provides frames as PNGs; the pipeline expects a video file.
    """
    # Find all frames in the directory
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith(DramaLayout.FRAME_PREFIX) and f.endswith(DramaLayout.FRAME_EXT)])
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    h, w, _ = first_frame.shape

    # Create a temporary file
    temp_dir = tempfile.gettempdir()
    temp_video_path = os.path.join(temp_dir, f"temp_{clip_id}.mp4")

    # Use a simple writer (10 fps as a standard for risk assessment epoch)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, 10.0, (w, h))

    for f_file in frame_files:
        img = cv2.imread(os.path.join(frames_dir, f_file))
        if img is not None:
            out.write(img)

    out.release()
    return temp_video_path

def run_benchmark_clip(
    clip_data: Dict[str, Any],
    dataset_root: str,
    config_override: Dict[str, Any],
    model_profile: str = None,
) -> Dict[str, Any]:
    """
    Runs the pipeline on a single DRAMA clip.
    """
    clip_id = clip_data["clip_id"]
    frames_dir = DramaLayout.get_clip_path(dataset_root, clip_id)

    # 1. Create temporary video
    temp_video_path = None
    original_config = copy.deepcopy(CONFIG)
    try:
        temp_video_path = create_temp_video(frames_dir, clip_id)

        # 2. Apply config overrides to global CONFIG
        # Note: This modifies global state. In a production runner, we might use a local context.
        # For this benchmark, we assume the runner is the main entry point.
        for section, values in config_override.items():
            if section not in CONFIG:
                CONFIG[section] = {}
            CONFIG[section].update(values)
        profile = apply_vlm_profile(model_profile)

        # 3. Initialize pipeline and process
        # Use cubic=False because split_panorama=False is handled in SegmentationPipeline.__init__
        pipeline = SegmentationPipeline(temp_video_path, cubic=False)

        # Process as a single epoch (one clip = one epoch)
        # We extract the results of process_epoch
        result = pipeline.process_epoch(
            frames=pipeline.video_loader.get_split_frames(max_to_extract=30),
            epoch_idx=0
        )

        return {
            "clip_id": clip_id,
            "processed": result["processed"],
            "reason": result.get("reason"),
            "risk_result": result.get("risk_result"),
            "environment_description": result.get("environment_description"),
            "segmented_items": result.get("segmented_items"),
            "environment_items": result.get("environment_items"),
            "model_profile": profile["id"],
            "model_repo_id": profile["repo_id"],
        }

    except Exception as e:
        logger.error(f"Failed to process clip {clip_id}: {e}")
        return {"clip_id": clip_id, "processed": False, "reason": str(e)}
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        CONFIG.clear()
        CONFIG.update(original_config)

def main():
    parser = argparse.ArgumentParser(description="DRAMA Benchmark Runner")
    parser.add_argument("--dataset-root", type=str, help="Path to DRAMA dataset")
    parser.add_argument("--config", type=str, default="benchmarks/drama/config_drama.yaml", help="Path to config override")
    parser.add_argument("--limit", type=int, help="Limit number of clips")
    parser.add_argument("--validate-dataset", action="store_true", help="Validate dataset and exit")
    parser.add_argument("--output-dir", type=str, default="outputs/drama_results", help="Output directory")
    parser.add_argument("--model", type=str, help="VLM profile ID from src/tasks/config/config.json (overrides configured selection).")

    args = parser.parse_args()

    # 1. Resolve dataset root (CLI > Env > Config)
    root = args.dataset_root or os.environ.get("DRAMA_ROOT")
    if root is None:
        # Try loading from config
        try:
            with open(args.config, "r") as f:
                import yaml
                cfg = yaml.safe_load(f) or {}
                root = cfg.get("dataset", {}).get("root")
        except Exception:
            pass

    # 2. Initialize Loader
    try:
        loader = DramaLoader(root)
    except ValueError as e:
        logger.error(e)
        exit(1)

    if args.validate_dataset:
        success, msg = loader.validate_dataset()
        print(msg)
        exit(0 if success else 1)

    # 3. Load Config Overrides
    config_override = {}
    try:
        with open(args.config, "r") as f:
            import yaml
            config_override = yaml.safe_load(f) or {}
            # Remove dataset root from override to avoid polluting pipeline CONFIG
            if "dataset" in config_override:
                del config_override["dataset"]
    except Exception as e:
        logger.warning(f"Could not load config override: {e}")

    profile_config = copy.deepcopy(CONFIG)
    for section, values in config_override.items():
        profile_config.setdefault(section, {}).update(values)
    model_profile = args.model or config_override.get("vlm", {}).get("selected_model")
    profile = resolve_vlm_profile(model_profile, profile_config)

    # 4. Load Clips
    annos = loader.load_annotations()
    if args.limit:
        annos = annos[:args.limit]

    # 5. Process Clips
    run_id = uuid.uuid4().hex
    run_name = f"{profile['id']}_{run_id}"
    run_output_dir = Path(args.output_dir) / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    with open(run_output_dir / "run_metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump({
            "run_id": run_id,
            "model_profile": profile["id"],
            "model_label": profile["label"],
            "model_repo_id": profile["repo_id"],
            "model_backend": profile["backend"],
            "dataset_root": root,
            "started_at": datetime.now().isoformat(),
        }, metadata_file, indent=2)
    results = []

    for i, clip_data in enumerate(annos):
        logger.info(f"[{i+1}/{len(annos)}] Processing clip {clip_data['clip_id']}...")
        res = run_benchmark_clip(clip_data, root, config_override, model_profile=profile["id"])
        res["run_id"] = run_id
        res["model_profile"] = profile["id"]
        res["model_repo_id"] = profile["repo_id"]
        results.append(res)

        # Save per-clip output for resumability
        with open(run_output_dir / f"{clip_data['clip_id']}.json", "w") as f:
            json.dump(res, f, indent=2)

    # 6. Final summary
    with open(run_output_dir / "all_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Benchmark complete. Results saved to {run_output_dir}")

if __name__ == "__main__":
    main()
