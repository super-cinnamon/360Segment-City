"""Render a full source video with the risk assessment timeline and overlays.

This script intentionally imports only standard-library modules, OpenCV, and
NumPy. It consumes the detailed risk JSONL, compressed per-epoch NPZ masks, and
the sensor CSV files next to the source video, so it can run after inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# DESIGN BLOCK: output format, layout proportions, typography, and palette.
# Edit these values to restyle the final video without changing data handling.
# The stacked layout uses a standard frame size for playback compatibility.
# ---------------------------------------------------------------------------
CANVAS_WIDTH = 2560
CANVAS_HEIGHT = 1440
HAZARD_PANEL_HEIGHT = 560
TELEMETRY_OVERLAY_HEIGHT = 150
PANEL_COLOR = (19, 24, 28)       # BGR: charcoal
TEXT_COLOR = (242, 245, 246)
MUTED_TEXT_COLOR = (159, 174, 181)
ACCENT_COLOR = (40, 210, 245)    # BGR: warm signal yellow
HAZARD_COLORS = (
    (52, 91, 246), (34, 157, 250), (35, 214, 230), (218, 83, 224),
    (235, 139, 53), (67, 207, 119), (209, 111, 66), (100, 180, 240),
)
FONT = cv2.FONT_HERSHEY_DUPLEX
OUTPUT_FPS_FALLBACK = 30.0
_PROJECTION_MAP_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


# ---------------------------------------------------------------------------
# DATA BLOCK: typed containers for one epoch and one displayable hazard.
# ---------------------------------------------------------------------------
@dataclass
class HazardCard:
    """One unique flagged object and the highest score/reason assigned to it."""

    object_id: str
    score: float
    class_name: str
    hazard_type: str
    observation: str
    explanation: str
    color: tuple[int, int, int]


@dataclass
class EpochRecord:
    """One processed or skipped epoch and its optional mask/telemetry sidecar."""

    epoch_idx: int
    raw_start: int
    raw_end: int
    result: dict[str, Any]
    sidecar_path: Optional[Path]
    processed: bool
    skip_reason: Optional[str]
    sidecar: Optional[dict[str, Any]] = None
    masks: Optional[dict[tuple[int, str, str], np.ndarray]] = None
    masks_by_frame_object_side: Optional[dict[tuple[int, str], list[tuple[str, np.ndarray]]]] = None


# ---------------------------------------------------------------------------
# INPUT BLOCK: parse CLI options and load the detailed JSONL records.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse paths and the quick-to-change threshold/output controls."""
    parser = argparse.ArgumentParser(
        description="Render risk scores, telemetry, and hazard masks over source footage."
    )
    parser.add_argument("--video", required=True, help="Replaceable original or censored source video.")
    parser.add_argument("--risk-data", required=True, help="Detailed risk results JSONL from the pipeline.")
    parser.add_argument("--output", default="outputs/risk_viz/hazard_analysis_full.mp4")
    parser.add_argument("--sidecar-dir", default="outputs/risk_viz", help="Directory containing per-epoch NPZ files.")
    parser.add_argument(
        "--telemetry-dir", default=None,
        help="Folder with 1_GPS5.csv, 1_ACCL.csv, 1_GYRO.csv, and 1_CORI.csv; defaults to the video folder.",
    )
    parser.add_argument(
        "--mask-threshold", type=float, default=0.0,
        help="Minimum expected risk score (1-5) for drawing masks and boxes; default 0 shows all.",
    )
    parser.add_argument("--width", type=int, default=CANVAS_WIDTH, help="Output width; default 2560.")
    parser.add_argument("--height", type=int, default=CANVAS_HEIGHT, help="Output height; default 1440.")
    parser.add_argument("--fps", type=float, default=None, help="Output frame rate; defaults to source FPS.")
    return parser.parse_args()


def read_risk_records(risk_path: Path, sidecar_dir: Path) -> list[EpochRecord]:
    """Read processed epochs and locate their sidecars without importing project code."""
    records = []
    with risk_path.open("r", encoding="utf-8") as risk_file:
        for line in risk_file:
            if not line.strip():
                continue
            entry = json.loads(line)
            meta = entry.get("meta", {})
            result = entry.get("result", {})
            epoch_idx = meta.get("epoch_idx")
            raw_start, raw_end = meta.get("raw_start"), meta.get("raw_end")
            if epoch_idx is None or raw_start is None or raw_end is None:
                continue

            processed = result.get("processed") is not False
            visual_path = result.get("viz_video_path") if processed else None
            sidecar_path = Path(visual_path).with_suffix(".npz") if visual_path else None
            if processed and (sidecar_path is None or not sidecar_path.exists()):
                sidecar_path = sidecar_dir / f"hazard_viz_{epoch_idx}.npz"
            records.append(EpochRecord(
                epoch_idx=int(epoch_idx),
                raw_start=int(raw_start),
                raw_end=int(raw_end),
                result=result,
                sidecar_path=(sidecar_path if sidecar_path is not None and sidecar_path.exists() else None),
                processed=processed,
                skip_reason=result.get("reason") if not processed else None,
            ))
    return sorted(records, key=lambda item: (item.raw_start, item.epoch_idx))


def load_epoch_sidecar(record: EpochRecord) -> None:
    """Load one epoch's JSON metadata and binary face masks into memory."""
    if record.sidecar_path is None:
        record.sidecar = {}
        record.masks = {}
        expected_ids = _agent_hazard_ids(record)
        if record.processed and expected_ids:
            raise ValueError(
                f"Epoch {record.epoch_idx} has hazards but no mask sidecar: "
                f"{', '.join(sorted(expected_ids))}"
            )
        return
    with np.load(record.sidecar_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        masks = {}
        for item in metadata.get("mask_entries", []):
            masks[(int(item["frame_index"]), item["object_id"], item["side"])] = archive[item["array_key"]].astype(bool)
    record.sidecar = metadata
    record.masks = masks
    expected_ids = _agent_hazard_ids(record)
    stored_ids = {entry["object_id"] for entry in metadata.get("mask_entries", [])}
    missing_ids = expected_ids - stored_ids
    if missing_ids:
        raise ValueError(
            f"Epoch {record.epoch_idx} hazard IDs have no saved masks: "
            f"{', '.join(sorted(missing_ids))}. Refusing to render a misleading video."
        )


def _agent_hazard_ids(record: EpochRecord) -> set[str]:
    """Return scored object IDs that should have segmentation-mask entries."""
    if not record.processed:
        return set()
    return {
        str(hazard.get("object_id"))
        for reason in record.result.get("reasons", [])
        for hazard in reason.get("hazards", [])
        if hazard.get("object_id") not in (None, "", "unknown", "environment")
    }


# ---------------------------------------------------------------------------
# TELEMETRY BLOCK: read the neighboring raw CSV streams without pandas or the
# application pipeline, then interpolate each available metric at video time.
# ---------------------------------------------------------------------------
def read_csv_track(
    directory: Path,
    filename: str,
    field_names: tuple[str, ...],
    transform=lambda value: value,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load a time/value pair from one sensor CSV, accepting t_s or cts time."""
    path = directory / filename
    if not path.is_file():
        return None
    timestamps = []
    values = []
    with path.open("r", encoding="utf-8-sig", newline="") as sensor_file:
        for row in csv.DictReader(sensor_file):
            try:
                time_value = row.get("t_s")
                if time_value in (None, ""):
                    time_value = float(row["cts"]) / 1000.0
                value = next(row[name] for name in field_names if row.get(name) not in (None, ""))
                timestamps.append(float(time_value))
                values.append(float(transform(float(value))))
            except (KeyError, StopIteration, TypeError, ValueError):
                continue
    if not timestamps:
        return None
    order = np.argsort(timestamps)
    return np.asarray(timestamps, dtype=np.float64)[order], np.asarray(values, dtype=np.float64)[order]


def load_direct_telemetry(directory: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load available GPS, accelerometer, gyroscope, and orientation CSV streams."""
    tracks = {}
    gps_speed = read_csv_track(
        directory, "1_GPS5.csv", ("GPS (2D speed) [m/s]",), transform=lambda value: value * 3.6
    )
    if gps_speed is not None:
        tracks["speed_kmh"] = gps_speed

    acceleration = read_csv_track(directory, "1_ACCL.csv", ("Accelerometer (x) [m/s²]",))
    if acceleration is not None:
        tracks["a_ego"] = acceleration

    yaw_rate = read_csv_track(
        directory, "1_GYRO.csv", ("Gyroscope (z) [rad/s]",), transform=math.degrees
    )
    if yaw_rate is not None:
        tracks["yaw_rate"] = yaw_rate

    quaternion_fields = (
        "CameraOrientation", "(1)", "(2)", "(3)",
    )
    orientation = read_csv_rows(directory / "1_CORI.csv", quaternion_fields)
    if orientation is not None:
        timestamps, columns = orientation
        qw, qx, qy, qz = columns
        roll = np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2)))
        tracks["roll"] = (timestamps, roll)
    return tracks


def read_csv_rows(
    path: Path,
    field_names: tuple[str, ...],
) -> Optional[tuple[np.ndarray, tuple[np.ndarray, ...]]]:
    """Load several named columns on one shared timestamp axis."""
    if not path.is_file():
        return None
    timestamps = []
    columns = [[] for _ in field_names]
    with path.open("r", encoding="utf-8-sig", newline="") as sensor_file:
        for row in csv.DictReader(sensor_file):
            try:
                time_value = row.get("t_s")
                if time_value in (None, ""):
                    time_value = float(row["cts"]) / 1000.0
                values = [float(row[name]) for name in field_names]
                timestamps.append(float(time_value))
                for column, value in zip(columns, values):
                    column.append(value)
            except (KeyError, TypeError, ValueError):
                continue
    if not timestamps:
        return None
    order = np.argsort(timestamps)
    return (
        np.asarray(timestamps, dtype=np.float64)[order],
        tuple(np.asarray(column, dtype=np.float64)[order] for column in columns),
    )


def telemetry_at_time(
    tracks: dict[str, tuple[np.ndarray, np.ndarray]],
    time_seconds: float,
) -> dict[str, Optional[float]]:
    """Interpolate available measured telemetry at a source-video timestamp."""
    values = {}
    for name, (timestamps, samples) in tracks.items():
        values[name] = float(np.interp(time_seconds, timestamps, samples))
    return values


# ---------------------------------------------------------------------------
# HAZARD BLOCK: merge reasons by object, sort by risk, and assign stable colors.
# ---------------------------------------------------------------------------
def expected_score(reason: dict[str, Any]) -> float:
    """Return the reason's 1-5 expected score, deriving it from probabilities if needed."""
    if reason.get("expected_score") is not None:
        return float(reason["expected_score"])
    probabilities = reason.get("score_probs", {})
    return sum(float(score) * float(probability) for score, probability in probabilities.items())


def build_hazard_cards(
    reasons: list[dict[str, Any]],
    class_names: Optional[dict[str, str]] = None,
) -> list[HazardCard]:
    """Deduplicate objects, using a reason-weighted score and strongest explanation."""
    selected: dict[str, dict[str, Any]] = {}
    for reason in reasons:
        score = expected_score(reason)
        weight = max(0.0, float(reason.get("softmax_weight", reason.get("weight", 1.0))))
        for hazard in reason.get("hazards", []):
            object_id = str(hazard.get("object_id", ""))
            if not object_id or object_id == "unknown":
                continue
            item = selected.setdefault(object_id, {
                "weighted_score_sum": 0.0,
                "weight_sum": 0.0,
                "score_sum": 0.0,
                "reason_count": 0,
                "explanation_weight": -1.0,
                "hazard": hazard,
            })
            item["weighted_score_sum"] += score * weight
            item["weight_sum"] += weight
            item["score_sum"] += score
            item["reason_count"] += 1
            if weight > item["explanation_weight"]:
                item["explanation_weight"] = weight
                item["hazard"] = hazard

    def object_score(item: dict[str, Any]) -> float:
        if item["weight_sum"] > 0:
            return item["weighted_score_sum"] / item["weight_sum"]
        return item["score_sum"] / item["reason_count"]

    ranked = sorted(selected.items(), key=lambda item: (-object_score(item[1]), item[0]))
    cards = []
    for index, (object_id, item) in enumerate(ranked):
        score = object_score(item)
        hazard = item["hazard"]
        cards.append(HazardCard(
            object_id=object_id,
            score=score,
            class_name=str(hazard.get("class_name") or (class_names or {}).get(object_id) or object_id),
            hazard_type=str(hazard.get("hazard_type") or "Hazard"),
            observation=str(hazard.get("observation") or "No observation text was logged."),
            explanation=str(hazard.get("danger_reasoning") or hazard.get("risk_critique") or "No explanation was logged."),
            color=HAZARD_COLORS[index % len(HAZARD_COLORS)],
        ))
    return cards


# ---------------------------------------------------------------------------
# PROJECTION BLOCK: inverse-map cube-face masks to the equirectangular source.
# The equations mirror src/tasks/preprocessing.py's generate_cubic() sampling.
# ---------------------------------------------------------------------------
def project_face_mask(
    mask: np.ndarray,
    side: str,
    source_height: int,
    source_width: int,
    overlap: float,
) -> np.ndarray:
    """Warp a face-space bool mask onto a source equirectangular frame."""
    face_height, face_width = mask.shape
    cache_key = (face_height, face_width, side, source_height, source_width, round(overlap, 6))
    maps = _PROJECTION_MAP_CACHE.get(cache_key)
    if maps is None:
        source_x, source_y = np.meshgrid(
            np.arange(source_width, dtype=np.float32),
            np.arange(source_height, dtype=np.float32),
        )
        longitude = (source_x / source_width - 0.5) * (2.0 * math.pi)
        latitude = (0.5 - source_y / source_height) * math.pi
        centers = {"front": 0.0, "right": math.pi / 2.0, "back": math.pi, "left": -math.pi / 2.0}
        delta = (longitude - centers[side] + math.pi) % (2.0 * math.pi) - math.pi

        # Each face's horizontal coordinate is tan(angle from its forward axis).
        face_u = np.tan(delta)
        tan_latitude = np.tan(latitude)
        denominator = np.maximum(1.0 - tan_latitude * tan_latitude, 1e-6)
        face_v = tan_latitude * np.sqrt((face_u * face_u + 1.0) / denominator)
        face_limit = 1.0 + max(0.0, overlap)
        horizontal_fov = math.atan(face_limit)
        valid = (
            (np.abs(delta) <= horizontal_fov)
            & (np.abs(face_u) <= face_limit)
            & (np.abs(face_v) <= face_limit)
        )

        # Left/right inputs were flipped during preprocessing; this inverse map
        # already addresses the flipped face coordinate, so do not flip again.
        map_x = ((face_u + face_limit) / (2.0 * face_limit) * (face_width - 1)).astype(np.float32)
        map_y = ((face_limit - face_v) / (2.0 * face_limit) * (face_height - 1)).astype(np.float32)
        maps = (map_x, map_y, valid)
        _PROJECTION_MAP_CACHE[cache_key] = maps
    map_x, map_y, valid = maps
    projected = cv2.remap(
        mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    projected[~valid] = False
    return projected


# ---------------------------------------------------------------------------
# LAYOUT BLOCK: reusable panel drawing, source fitting, and hazard annotations.
# Keep visual edits in these functions and the DESIGN BLOCK above.
# ---------------------------------------------------------------------------
def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = TEXT_COLOR,
    thickness: int = 1,
) -> None:
    """Draw one anti-aliased line of text in BGR canvas coordinates."""
    cv2.putText(canvas, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def fit_source_frame(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, int, int]:
    """Aspect-fit source footage into a panel and return image plus top-left offset."""
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    fitted_width = max(1, int(source_width * scale))
    fitted_height = max(1, int(source_height * scale))
    fitted = cv2.resize(frame, (fitted_width, fitted_height), interpolation=cv2.INTER_AREA)
    offset_x = (width - fitted_width) // 2
    offset_y = (height - fitted_height) // 2
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[offset_y:offset_y + fitted_height, offset_x:offset_x + fitted_width] = fitted
    return panel, offset_x, offset_y


def wrap_words(text: str, max_characters: int) -> list[str]:
    """Wrap a description into predictable, renderer-friendly text rows."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_characters and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def mask_bounding_box(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Return an exclusive (x0, y0, x1, y1) box for a 2D mask."""
    y_coordinates, x_coordinates = np.where(mask)
    if x_coordinates.size == 0:
        return None
    return (
        int(x_coordinates.min()),
        int(y_coordinates.min()),
        int(x_coordinates.max()) + 1,
        int(y_coordinates.max()) + 1,
    )


def draw_telemetry_overlay(
    video_panel: np.ndarray,
    telemetry: dict[str, Optional[float]],
) -> None:
    """Draw only available live values over transparent shading on the video."""
    metrics = (
        ("SPEED", "speed_kmh", "km/h"),
        ("ACCELERATION", "a_ego", "m/s2"),
        ("ROLL", "roll", "deg"),
        ("YAW RATE", "yaw_rate", "deg/s"),
    )
    overlay_height = min(TELEMETRY_OVERLAY_HEIGHT, video_panel.shape[0])
    shade = video_panel[:overlay_height].copy()
    shade[:] = PANEL_COLOR
    cv2.addWeighted(shade, 0.62, video_panel[:overlay_height], 0.38, 0,
                    video_panel[:overlay_height])

    column_width = video_panel.shape[1] // len(metrics)
    for index, (label, key, unit) in enumerate(metrics):
        x = 42 + index * column_width
        y = 56
        value = telemetry.get(key)
        rendered = "--" if value is None else f"{float(value):.1f} {unit}"
        draw_text(video_panel, label, (x, y), 0.62, MUTED_TEXT_COLOR, 1)
        draw_text(video_panel, rendered, (x, y + 48), 1.05, TEXT_COLOR, 2)


def draw_hazard_panel(
    canvas: np.ndarray,
    cards: list[HazardCard],
    threshold: float,
    skipped_reason: Optional[str] = None,
) -> None:
    """Draw score-ranked hazard descriptions in the bottom log band."""
    panel_top = canvas.shape[0] - HAZARD_PANEL_HEIGHT
    canvas[panel_top:] = PANEL_COLOR
    cv2.line(canvas, (0, panel_top), (canvas.shape[1], panel_top), ACCENT_COLOR, 2)
    draw_text(canvas, "HAZARD LOG", (42, panel_top + 48), 0.88, TEXT_COLOR, 2)
    draw_text(canvas, f"MASK THRESHOLD {threshold:.2f}/5",
              (canvas.shape[1] - 390, panel_top + 46), 0.55, MUTED_TEXT_COLOR, 1)

    if skipped_reason:
        draw_text(canvas, "Risk assessment skipped", (42, panel_top + 120), 0.74, ACCENT_COLOR, 1)
        draw_text(canvas, f"Reason: {skipped_reason}", (42, panel_top + 170), 0.60, MUTED_TEXT_COLOR, 1)
        return
    if not cards:
        draw_text(canvas, "No hazard reasons were recorded for this epoch.",
                  (42, panel_top + 120), 0.64, MUTED_TEXT_COLOR, 1)
        return

    list_top = panel_top + 80
    columns = 2
    row_count = (len(cards) + columns - 1) // columns
    available_height = HAZARD_PANEL_HEIGHT - 90
    row_height = available_height // row_count
    column_width = canvas.shape[1] // columns
    max_characters = max(50, int((column_width - 115) / 9))
    for index, card in enumerate(cards):
        row, column = divmod(index, columns)
        x = column * column_width + 42
        y = list_top + row * row_height
        if y + 36 > canvas.shape[0]:
            continue
        cv2.rectangle(canvas, (x, y - 17), (x + 17, y + 4), card.color, -1)
        draw_text(canvas, f"{card.score:.2f}/5", (x + 28, y), 0.65, card.color, 2)
        title = f"{card.class_name} | {card.hazard_type} | {card.object_id}"
        draw_text(canvas, title[:max_characters], (x + 128, y), 0.61, TEXT_COLOR, 1)
        detail = f"{card.observation} Why flagged: {card.explanation}"
        detail_lines = wrap_words(detail, max_characters)
        detail_x = x + 28
        detail_y = y + 27
        for line_index, line in enumerate(detail_lines[:2]):
            draw_text(canvas, line, (detail_x, detail_y + line_index * 22),
                      0.49, MUTED_TEXT_COLOR, 1)
        if card.score < threshold:
            draw_text(canvas, "BELOW THRESHOLD", (x + column_width - 270, y),
                      0.48, MUTED_TEXT_COLOR, 1)


def draw_hazard_overlays(
    video_panel: np.ndarray,
    source_frame: np.ndarray,
    record: EpochRecord,
    frame_in_epoch: int,
    cards: list[HazardCard],
    threshold: float,
) -> None:
    """Draw eligible masks and risk boxes at the final display resolution."""
    video_height, video_width = video_panel.shape[:2]
    source_height, source_width = source_frame.shape[:2]
    display_scale = min(video_width / source_width, video_height / source_height)
    fitted_width = max(1, int(source_width * display_scale))
    fitted_height = max(1, int(source_height * display_scale))
    offset_x = (video_width - fitted_width) // 2
    offset_y = (video_height - fitted_height) // 2

    for card in cards:
        if card.score < threshold:
            continue
        face_masks = (
            record.masks_by_frame_object_side.get((frame_in_epoch, card.object_id), [])
            if record.masks_by_frame_object_side else []
        )
        if not face_masks:
            continue
        source_mask = np.zeros((fitted_height, fitted_width), dtype=bool)
        overlap = float((record.sidecar or {}).get("side_overlap", 0.0))
        for side, mask in face_masks:
            source_mask |= project_face_mask(mask, side, fitted_height, fitted_width, overlap)
        if not np.any(source_mask):
            continue

        bounds = mask_bounding_box(source_mask)
        if bounds is None:
            continue
        local_x0, local_y0, local_x1, local_y1 = bounds
        x0, x1 = offset_x + local_x0, offset_x + local_x1
        y0, y1 = offset_y + local_y0, offset_y + local_y1
        image_region = video_panel[offset_y:offset_y + fitted_height, offset_x:offset_x + fitted_width]
        image_region[source_mask] = (
            0.58 * image_region[source_mask] + 0.42 * np.asarray(card.color)
        ).astype(np.uint8)
        cv2.rectangle(video_panel, (x0, y0), (x1 - 1, y1 - 1), card.color, 3, cv2.LINE_AA)
        label = f"{card.class_name}  RISK {card.score:.2f}/5"
        label_y = max(36, y0 - 12)
        cv2.rectangle(video_panel, (x0, label_y - 36),
                      (min(video_width - 1, x0 + min(700, len(label) * 19)), label_y + 8),
                      card.color, -1)
        draw_text(video_panel, label, (x0 + 10, label_y), 0.78, (255, 255, 255), 2)


# ---------------------------------------------------------------------------
# VIDEO BLOCK: load data in chronological order and render the full source clip.
# ---------------------------------------------------------------------------
def render_video(args: argparse.Namespace) -> None:
    """Render source footage with translucent telemetry and a bottom hazard log."""
    video_path = Path(args.video)
    risk_path = Path(args.risk_data)
    output_path = Path(args.output)
    if not video_path.is_file():
        raise FileNotFoundError(f"Source video not found: {video_path}")
    if not risk_path.is_file():
        raise FileNotFoundError(f"Risk JSONL not found: {risk_path}")
    width, height = int(args.width), int(args.height)
    if width < 1280 or height < 720 or width % 2 or height % 2:
        raise ValueError("Output dimensions must be even and at least 1280x720.")
    video_height = height - HAZARD_PANEL_HEIGHT
    if video_height < 240:
        raise ValueError("The selected output height leaves too little space for the video panel.")

    records = read_risk_records(risk_path, Path(args.sidecar_dir))
    if not records:
        raise ValueError("No epochs with raw_start/raw_end metadata were found.")

    telemetry_dir = Path(args.telemetry_dir) if args.telemetry_dir else video_path.parent
    telemetry_tracks = load_direct_telemetry(telemetry_dir)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or OUTPUT_FPS_FALLBACK
    output_fps = float(args.fps or source_fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video writer: {output_path}")

    record_index = 0
    current_record: Optional[EpochRecord] = None
    source_frame_index = 0
    try:
        while True:
            success, source_frame = capture.read()
            if not success:
                break

            while record_index < len(records) and source_frame_index >= records[record_index].raw_end:
                record_index += 1
                current_record = None
            if record_index < len(records):
                candidate = records[record_index]
                if candidate.raw_start <= source_frame_index < candidate.raw_end:
                    if current_record is not candidate:
                        load_epoch_sidecar(candidate)
                        candidate.masks_by_frame_object_side = {}
                        for item in (candidate.sidecar or {}).get("mask_entries", []):
                            key = (int(item["frame_index"]), item["object_id"])
                            candidate.masks_by_frame_object_side.setdefault(key, []).append(
                                (item["side"], candidate.masks[(key[0], key[1], item["side"])])
                            )
                        current_record = candidate

            skipped_reason = None
            cards = []
            if current_record:
                frame_in_epoch = source_frame_index - current_record.raw_start
                skipped_reason = current_record.skip_reason
                class_names = {
                    object_id: metadata.get("class_name", object_id)
                    for object_id, metadata in (current_record.sidecar or {}).get("hazards", {}).items()
                }
                if current_record.processed:
                    cards = build_hazard_cards(current_record.result.get("reasons", []), class_names)

            # Telemetry is sampled from raw CSVs for every source frame, including
            # skipped epochs and any gaps between recorded epoch windows.
            telemetry = telemetry_at_time(telemetry_tracks, source_frame_index / source_fps)
            canvas = np.full((height, width, 3), PANEL_COLOR, dtype=np.uint8)
            video_panel, _, _ = fit_source_frame(source_frame, width, video_height)
            if current_record and current_record.processed:
                draw_hazard_overlays(
                    video_panel, source_frame, current_record, frame_in_epoch,
                    cards, args.mask_threshold,
                )
            draw_telemetry_overlay(video_panel, telemetry)
            canvas[:video_height] = video_panel
            draw_hazard_panel(canvas, cards, args.mask_threshold, skipped_reason)
            writer.write(canvas)
            source_frame_index += 1
            if source_frame_index % 500 == 0:
                print(f"Rendered {source_frame_index} source frames...", flush=True)
    finally:
        capture.release()
        writer.release()

    print(f"Rendered {source_frame_index} frames at {width}x{height} to {output_path}")


def main() -> None:
    """CLI entry point; threshold defaults to zero to display every flagged agent."""
    render_video(parse_args())


if __name__ == "__main__":
    main()