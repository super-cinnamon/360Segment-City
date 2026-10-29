# 360Segment-City

360Segment-City is a computer-vision pipeline for analyzing 360-degree motorcycle
video and estimating road-user and scene risk. It combines semantic segmentation,
depth estimation, optical flow, object tracking, optional GoPro telemetry, and a
local vision-language model (VLM) to produce per-epoch risk assessments.

## Processing overview

For each video window, the pipeline can split an equirectangular panorama into
front, back, left, and right views. It then detects and tracks road users,
estimates their depth and motion, and combines those observations with scene
context and optional vehicle telemetry. Risk results are written as JSON Lines;
visualization videos are generated for processed epochs.

## Requirements and setup

Use Python with versions of PyTorch and the model libraries compatible with your
machine. A CUDA-capable GPU is strongly recommended: the vision models and VLM
are large, and CPU inference can be very slow. PyTorch selects CUDA when it is
available and otherwise runs on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install fastapi uvicorn openai scipy pytest
```

The extra packages provide runtime imports used by the API, risk assessment,
depth estimation, and tests. Install a PyTorch build appropriate for your CUDA
version if the default `torch` installation does not match your system. The
first run may download model weights from Hugging Face or Ultralytics; make sure
the model repositories are accessible and that you have enough disk space and
GPU memory.

The full runner expects the StrongSORT checkpoint at
`external_tracking/deep_sort/deep/checkpoint/ckpt.t7`. If it is absent, the
runner prints the download command; tracking may be degraded without it.

## Input data

The main runner accepts either a directory or a video path. For telemetry-aware
processing, pass a directory containing `360_video.mp4` and, when available,
GoPro GPMF CSV exports:

```text
data/sample_1/
├── 360_video.mp4
├── 1_ACCL.csv   # linear acceleration (optional)
├── 1_CORI.csv   # camera orientation (optional)
├── 1_GPS5.csv   # GPS and speed (optional)
├── 1_GRAV.csv   # gravity vector (optional)
├── 1_GYRO.csv   # angular velocity (optional)
└── 1_IORI.csv   # image orientation (optional)
```

Telemetry is enabled by default in the configuration. Missing or invalid CSV
streams are skipped, and risk processing can continue without telemetry. The
CSV exports should include a `t_s` time column or a `cts` column in milliseconds.
Passing a video file directly disables telemetry processing.

## Run the pipeline

From the repository root, run the full epoch-based pipeline against a data
directory:

```bash
python run_full_process.py --sample_dir data/sample_1
```

Add `--debug` for debug-level logging. The runner starts the local API on
`127.0.0.1:5015`; keep that port available. Processing starts after the API
initialization delay and may take substantial time depending on video length,
hardware, and model choices.

The full runner creates a timestamped run folder to keep its risk records and
mask sidecars aligned. It automatically renders the final analysis video after
the framework finishes. For a sample directory named `sample_1`, outputs are:

- `data/outputs/sample_1/<run_id>/paper_risk_epochs_detailed.jsonl` — per-epoch
  results and skip reasons.
- `data/outputs/sample_1/<run_id>/paper_risk_epochs_summary.jsonl` — per-epoch
  final risk scores.
- `outputs/risk_viz/sample_1/<run_id>/hazard_viz_<epoch>.npz` — the masks and
  metadata corresponding to that exact run's hazard IDs.
- `outputs/risk_viz/sample_1/<run_id>/hazard_analysis_full.mp4` — automatically
  rendered full source video with telemetry and hazard overlays.
- `outputs/risk_viz/sample_1/<run_id>/hazard_viz_full.mp4` — merged per-epoch
  visualization clips.

Each run gets a new unique folder, so older logs and masks are not reused or
overwritten.

To render the original source video with larger telemetry above it and scored
hazard overlays/reasons below it, run the independent compositor after processing:

```bash
python scripts/render_hazard_video.py \
  --video data/sample_1/360_video.mp4 \
  --risk-data data/outputs/sample_1/<run_id>/paper_risk_epochs_detailed.jsonl \
  --sidecar-dir outputs/risk_viz/sample_1/<run_id> \
  --telemetry-dir data/sample_1 \
  --output outputs/risk_viz/sample_1/<run_id>/hazard_analysis_full.mp4 \
  --mask-threshold 0
```

Change `--video` to a frame-aligned censored replacement when needed. The
output uses a standard 2560x1440 MP4 frame, with live telemetry translucently
overlaid at the top of the source video and scored hazard logs at the bottom.
The telemetry overlay shows speed, acceleration, roll, and yaw rate from the
available CSVs. Use `--width` and `--height` to adjust the output resolution.
`--telemetry-dir` points to
the folder containing the sensor CSVs and defaults to the source video's
folder. `--mask-threshold` is an expected risk score on the 1-5 scale; zero
displays every flagged agent. The replacement video must retain the same frame
timing and dimensions as the source used for inference so saved masks remain
aligned.

## Configuration

Pipeline settings are centralized in
[`src/tasks/config/config.json`](src/tasks/config/config.json). The Python
configuration loader reads this file at startup.

| Section | Purpose |
| --- | --- |
| `depth_estimation` | Selects Marigold or YOLO depth and configures each model. |
| `segmentation` | Configures the OneFormer model, class labels, and thresholds. |
| `flow_estimation` | Selects the optical-flow model. |
| `vlm` | Selects a VLM profile and configures model backends, repositories, and sampling settings. |
| `processing` | Controls panorama views, frame sizing, ROI gating, frame pruning, and telemetry. |
| `risk_assessment` | Sets the local API URL, model, epoch window, and score parameters. |
| `risk_map` | Controls risk-map model, weights, and hyperparameters. |
| `fragment_merging` | Configures merging of fragmented detections. |

The world-model API is configured for `http://localhost:5015/v1` and is started
by `run_full_process.py`. VLM profiles are listed under
`vlm.models`; `vlm.selected_model` chooses the profile used by default. Changing
the selection affects only that run/profile; the other model backends remain
available for quick switching.

Each profile declares a `repo_id`, `backend`, and video capability. Set a
candidate's exact Hugging Face repository in `repo_id` and verify that the
checkpoint accepts the pipeline's video inputs before setting
`supports_video` to `true`. Qwen 3.8 and both Gemma 4 profiles use the
in-process Unsloth `FastVisionModel` loader. Nemotron 3 Nano Omni is a GGUF
profile served by a separate `llama-server` process; install a llama.cpp build
with multimodal support and make `llama-server` available on `PATH` before
selecting it. Download its configured model and projector files with:

```bash
python scripts/download_vlm_models.py --model nemotron-3-nano-omni-30b-a3b-reasoning
```

The local API starts the GGUF server only when that profile is selected. Other
profiles continue to use their configured in-process backend. The text-only
Nemotron Bolt model is not in the selectable catalog.
Profiles using the `unsloth` backend load Unsloth lazily; install its runtime
for the active PyTorch/CUDA environment before selecting them. The Cosmos
profile uses Transformers and does not require Unsloth.

Use the config for a persistent default or `--model` for a one-run override;
the CLI value does not rewrite `config.json`:

```bash
python run_full_process.py --sample_dir data/sample_1
python run_full_process.py --sample_dir data/sample_1 --model cosmos3-nano
```

Run outputs are placed under `data/outputs/<sample>/<profile>_<uuid>/`, with
matching visualization directories and a `run_metadata.json` recording the
profile, repository, backend, and run ID.

Download configured profile files into the Hugging Face disk cache without
loading model weights into RAM or GPU:

```bash
python scripts/download_vlm_models.py
python scripts/download_vlm_models.py --model gemma-4-26b-a4b --cache-dir /path/to/hf-cache
```

The Nano Omni profile downloads only its Q4_K_M GGUF and F16 vision projector,
not every quantization in the upstream repository. GGUF inference still
requires a compatible serving backend; the current local Transformers/Unsloth
loader reports this profile as unsupported for pipeline inference.

## Depth estimation

Depth estimation is selected with `depth_estimation.mode` in
[`src/tasks/config/config.json`](src/tasks/config/config.json):

- `"marigold"` (default) keeps the existing Marigold pipeline and output unchanged.
- `"yolo"` uses the configured Ultralytics model in
  `depth_estimation.models.yolo.model_name` (default: `yolo26x-depth.pt`).

Model-specific settings live under `depth_estimation.models`: Marigold's model,
pipeline, and hyperparameters are under `marigold`, while YOLO's model name is
under `yolo`.

Both modes return a Spectral-colored PIL image and a float depth map matching the
input frame's height and width. YOLO's depth map is normalized per frame to
`[0, 1]` for the existing relative-depth processing; these values are relative,
not metric distances. Test-mode depth caches are separate: Marigold uses
`depth_...` cache names, while YOLO uses `depth_yolo_...`.

## Tests

Run the maintained test suite with:

```bash
python -m pytest tests/
```

Manual probes and test-oriented utilities live in
[`tests/manual/README.md`](tests/manual/README.md). They are intentionally
excluded from routine pytest collection because many load large models or
require local video and cache files.
