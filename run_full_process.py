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


def build_env_description_for_frames(pipeline, frames):
    # For cubic pipelines, convert to cubic and take front faces
    if pipeline.video_processor.cubic:
        cubic = pipeline.video_loader.generate_cubic(frames)
        front_frames = cubic["front"]
    else:
        front_frames = frames

    env_chunks = []
    for i in range(0, len(front_frames), 10):
        desc = query_world_model(
            prompt=ENV_PROMPT,
            images=front_frames[i:i+10],
            model=pipeline.video_loader.__class__.__module__ and None,
        )
        env_chunks.append(desc)

    # Normalise to string
    last = env_chunks[-1] if env_chunks else ""
    if isinstance(last, list):
        return "\n".join(last)
    return last


def main():
    ensure_out_dir()

    # Start API server in background (same as run_process.py)
    start_api_in_background()

    pipeline = SegmentationPipeline(DATA_PATH, cubic=True)

    start_raw = 0
    window_idx = 0

    while True:
        frames, last_raw = pipeline.video_loader.get_frames_window(start_raw, max_to_extract=CHUNK_MAX_FRAMES)
        if not frames:
            print("No more frames to process. Exiting.")
            break

        print(f"Processing window {window_idx} raw_frames {start_raw}..{last_raw} (unique frames={len(frames)})")

        # Process vision for this frame window
        segmented_items = pipeline.process_vision_for(frames)

        # Build environment description for the window
        if pipeline.video_processor.cubic:
            cubic = pipeline.video_loader.generate_cubic(frames)
            front_frames = cubic["front"]
        else:
            front_frames = frames

        # Query environment on front frames in groups of 10
        env_description = ""
        for i in range(0, len(front_frames), 10):
            env_part = query_world_model(
                prompt=ENV_PROMPT,
                images=front_frames[i:i+10],
                model=CONFIG["vlm"]["world_model"]["model_name"],
            )
            if isinstance(env_part, list):
                env_part = "\n".join(env_part)
            env_description += env_part + "\n"

        env_description = env_description.strip()

        # Assess risk for this window
        epoch_results = pipeline.process_risk(segmented_items, env_description)

        # Normalise epoch_results to a list (process_risk may return dict or list)
        results_list = epoch_results if isinstance(epoch_results, list) else [epoch_results]

        # Save detailed and summary outputs per epoch
        for idx, res in enumerate(results_list):
            meta = {
                "window_idx": window_idx,
                "raw_start": start_raw,
                "raw_end": last_raw,
                "epoch_in_window": idx,
                "epoch_frames": res.get("context_summary", {}).get("epoch_frames", None),
            }
            detailed = {"meta": meta, "result": res}
            save_jsonl(DETAILED_OUT, detailed)

            summary = {"meta": meta, "final_score": res.get("final_score", res.get("expected_risk_score", None))}
            save_jsonl(SUMMARY_OUT, summary)

        # Clean up large objects to free RAM/VRAM
        del frames
        try:
            del cubic
        except Exception:
            pass
        del segmented_items
        del epoch_results
        del results_list
        gc.collect()

        # advance to next raw frame after the last read
        if last_raw is None:  # * if you'd like to process only one window (epoch), force this break by uncommenting the comment after it
            break

        # break

        # * if you'd like to run k epochs, change the k value and uncomment this comment
        # k = 5
        # if window_idx == k:
        #     break

        start_raw = last_raw + 1
        window_idx += 1

    print("Processing complete. Outputs written:")
    print(DETAILED_OUT)
    print(SUMMARY_OUT)


if __name__ == "__main__":
    main()
