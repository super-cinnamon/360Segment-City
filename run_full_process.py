import os
import json
import gc

from src.process import SegmentationPipeline
from src.tasks.api.api_run import start_api_in_background
from src.tasks.environment import query_world_model
from src.tasks.config.utils import ENV_PROMPT, CONFIG

DATA_PATH = "data/GS010013.mp4"
CHUNK_MAX_FRAMES = 2000
OUTPUT_DIR = "data/outputs"
DETAILED_OUT = os.path.join(OUTPUT_DIR, "risk_epochs_detailed.jsonl")
SUMMARY_OUT = os.path.join(OUTPUT_DIR, "risk_epochs_summary.jsonl")


def ensure_out_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def main():
    ensure_out_dir()

    # Start API server in background (same as run_process.py)
    start_api_in_background()

    pipeline = SegmentationPipeline(DATA_PATH, cubic=True)

    epoch_size = CONFIG["risk_assessment"].get("epoch_window_size", 50)
    start_raw = 0
    epoch_idx = 0

    while True:
        # Load a large chunk of frames to minimize I/O and RAM overhead
        frames, last_raw = pipeline.video_loader.get_frames_window(start_raw, max_to_extract=CHUNK_MAX_FRAMES)
        if not frames:
            print("No more frames to process. Exiting.")
            break

        print(f"Processing chunk raw_frames {start_raw}..{last_raw} (unique frames={len(frames)})")

        # Process the chunk in epoch-sized windows
        for i in range(0, len(frames), epoch_size):
            epoch_frames = frames[i : i + epoch_size]

            # Perform ROI gating, environment description, and risk assessment per epoch
            result = pipeline.process_epoch(
                epoch_frames,
                roi_enabled=CONFIG.get("processing", {}).get("roi_enabled", True),
                roi_threshold=CONFIG.get("processing", {}).get("roi_threshold", 0.75),
            )

            # Calculate actual raw frame indices for this epoch
            actual_start = start_raw + i
            actual_end = start_raw + min(i + epoch_size, len(frames))

            meta = {
                "epoch_idx": epoch_idx,
                "raw_start": actual_start,
                "raw_end": actual_end,
                "epoch_frames": result.get("risk_result", {}).get("context_summary", {}).get("epoch_frames", None) if result.get("processed") else None,
            }

            if result["processed"]:
                res = result["risk_result"]
                save_jsonl(DETAILED_OUT, {"meta": meta, "result": res})
                save_jsonl(SUMMARY_OUT, {"meta": meta, "final_score": res.get("final_score", res.get("expected_risk_score", None))})
            else:
                # Save skipped epoch with the reason (e.g., "skip_low_roi")
                save_jsonl(DETAILED_OUT, {"meta": meta, "result": {"processed": False, "reason": result["reason"]}})

            epoch_idx += 1

        # Clean up large objects to free RAM/VRAM
        del frames
        gc.collect()

        # Advance to the next raw frame after the last read
        if last_raw is None:
            break

        start_raw = last_raw + 1

    print("Processing complete. Outputs written:")
    print(DETAILED_OUT)
    print(SUMMARY_OUT)


if __name__ == "__main__":
    main()
