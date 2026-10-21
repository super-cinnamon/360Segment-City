"""
risk_assessment.py
==================
Implements the G-VEval framework (arXiv 2412.13647, building on G-Eval Liu et al.
2023) adapted for 360° motorcycle road-safety risk assessment.

Pipeline per frame
------------------
  Step 1 — Reason sampling  (N independent calls, temperature > 0)
      The model receives a Task Introduction and Evaluation Criteria.  It is
      called N times independently to produce N distinct "reason" interpretations
      of the scene.  Each call returns both the reason text and the per-token
      log-probabilities used to estimate P(rᵢ).

  Step 2 — Synthesized Scoring (Smarter Fusion)
      Instead of N independent scoring calls, a single synthesis call evaluates
      all N reasons, selects the most grounded, and produces a final risk score
      distribution via log-probabilities.

  Step 3 — Final Score Calculation
      The final risk score is the expectation over the synthesized distribution.

References
----------
  Liu et al. (2023) "G-Eval: NLG Evaluation using GPT-4 with Better Human
  Alignment."  ACL 2023.  https://aclanthology.org/2023.acl-long.244

  G-VEval (2024) "Fine-Grained Risk Assessment."
  arXiv 2412.13647  https://arxiv.org/pdf/2412.13647
"""

import re
import json
import math
import hashlib
import logging
import cv2
import base64
import numpy as np
import os
from pathlib import Path
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Optional, Union

from src.tasks.risk_map import RiskMapEngine, Hazard
from src.tasks.config.utils import CONFIG
from src.tasks.fragment_merger import FragmentMerger

# Module logger
logger = logging.getLogger(__name__)

# Debug directory for WM inputs
DEBUG_WM_DIR = Path("outputs/debug_wm_inputs")
DEBUG_WM_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Module-level config aliases
# ---------------------------------------------------------------------------
_RA_CFG           = CONFIG["risk_assessment"]
_DEFAULT_API_BASE = _RA_CFG["api_base"]
_DEFAULT_MODEL    = _RA_CFG["model_name"]
_DEFAULT_N        = int(_RA_CFG.get("n_reasons", 5))
_NORMALIZE_SCORES = _RA_CFG.get("normalize_scores", True)
_SOFTMAX_TEMP     = float(_RA_CFG.get("softmax_temperature", 0.7))
_SIDE_OVERLAP_MATCHER = FragmentMerger()

# ---------------------------------------------------------------------------
# G-Eval Evaluation Criteria
# ---------------------------------------------------------------------------
EVALUATION_CRITERIA: list[tuple[str, str]] = [
    (
        "Agent Proximity & Count",
        "How many dynamic agents are present and how close are they?  "
        "Proximity weight 100% means the object is directly in front / "
        "at the camera plane; 0% means it is at the far depth limit.  "
        "Multiple high-proximity agents compound the risk.",
    ),
    (
        "Agent Type & Vulnerability",
        "What category of road users are detected?  Pedestrians, cyclists, "
        "and riders are more vulnerable and less predictable.  Specific "
        "attention must be paid to children, who exhibit highly unpredictable "
        "behavior. Their presence at high proximity significantly raises risk.",
    ),
    (
        "Intent & Predictability",
        "Are there indicators of movement or intent?  Look for road users facing "
        "the road (ready to enter), active turn signals, or erratic trajectories. "
        "Lack of clear intent or conflicting signals increases risk.",
    ),
    (
        "Directional Exposure",
        "Are hazards distributed across multiple camera sides (front, left, "
        "right, back)?  A 360° threat is more severe than a single-axis one.",
    ),
    (
        "Road & Environmental Conditions",
        "What do the environment features indicate about road surface, weather, "
        "lighting, traffic density, and lane structure?  Adverse conditions "
        "(wet surface, night, heavy traffic, fog) multiply effective risk.",
    ),
    (
        "Physical Telemetry (if available)",
        "If telemetry data is present: is the Time-To-Collision (TTC) short? "
        "Is relative velocity high?  Is the lateral clearance small?  Is the "
        "motorcycle rolling or yawing rapidly?  Each of these independently "
        "elevates risk.  If telemetry is absent, skip this criterion.",
    ),
    (
        "Rider Awareness & Attention",
        "Assess the likelihood that the rider is aware of the hazard. "
        "Risk is significantly higher if a hazard is closing in from a blind spot, "
        "from an unexpected angle, or if the rider's trajectory and speed suggest a "
        "lack of defensive response to an approaching agent.",
    ),
    (
        "Signage & Signal Context",
        "Integrate the state of traffic signals and regulatory signs. "
        "A red light, stop sign, or restrictive marking increases the risk of "
        "unexpected agent movements or requires immediate rider action.",
    ),
    (
        "Lane Positioning & Geometry",
        "Evaluate the risk associated with the rider's position in the lane and the "
        "lane's geometry. Riding on lane edges in high-speed traffic, merging "
        "into narrowing lanes, or violating solid markings elevates risk.",
    ),
]

# ---------------------------------------------------------------------------
# Score rubric — shared between reason-generation and scoring prompts
# ---------------------------------------------------------------------------
SCORE_RUBRIC = (
    "1 = MINIMAL RISK  — Clear road; all agents distant; no action required.\n"
    "2 = LOW RISK      — Light traffic nearby; standard defensive riding applies.\n"
    "3 = MODERATE RISK — Elevated caution required; one or more agents are close "
    "or behaving unpredictably; a speed or position adjustment may be needed.\n"
    "4 = HIGH RISK     — Significant hazard; braking or evasive manoeuvre likely "
    "required soon; collision window is opening.\n"
    "5 = SEVERE RISK   — Imminent hazard; emergency manoeuvre required right now "
    "or collision is unavoidable."
)


# ---------------------------------------------------------------------------
# 1. Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TelemetryData:
    """
    Optional physics-sensor telemetry (e.g. from IMUs).
    Pass None where TelemetryData is accepted to omit the telemetry section
    of the G-VEval prompt — the model will skip the telemetry criterion.

    Fields
    ------
    ttc       : Time To Collision (seconds)        — derived from sensor fusion
    pet       : Post Encroachment Time (seconds)   — derived from sensor fusion
    d_lat     : Lateral distance to nearest agent (metres)
    v_rel     : Relative speed to nearest agent (m/s)
    speed_ms    : Ego-motorcycle speed (m/s)
    speed_kmh   : Ego-motorcycle speed (km/h)
    a_ego     : Ego-motorcycle longitudinal acceleration (m/s²) — from IMU
    roll      : Roll angle (degrees)               — from IMU
    yaw_rate  : Yaw rate (deg/s)                   — from IMU
    """
    ttc:      float
    pet:      float
    d_lat:    float
    v_rel:    float
    speed_ms:   float
    speed_kmh:  float
    a_ego:    float
    roll:     float
    yaw_rate: float


@dataclass
class DetectedAgent:
    """
    A single segmented road object for one camera side / frame, converted
    from the dicts produced by SegmentationPipeline.process_vision().

    Segment dict schema (source: process.py)
    -----------------------------------------
    {
        "frame":      int,
        "side":       str,          # "front" | "left" | "right" | "back"
        "class_name": str,
        "class_id":   int,
        "instance_id": int,         # instance id in the segmentation map
        "score":      float | None, # segmentation confidence
        "was_fused":  bool,
        "mask":       np.ndarray,   # binary boolean mask (H × W)
        "mode_depth": float,        # modal depth; LOWER = CLOSER = HIGHER RISK
    }

    distance_weight
    ---------------
    Derived from mode_depth via global-batch normalisation:
        distance_weight = 1 − (mode_depth − global_min) / (global_max − global_min)
    Range [0.0, 1.0].  1.0 = closest object in the batch; 0.0 = furthest.
    """
    agent_id:        str
    class_name:      str
    side:            str
    frame:           int
    seg_score:       float
    mode_depth:      float
    distance_weight: float
    instance_id:     int
    fused_sides:     list[str] = field(default_factory=list)
    masks:           dict[str, np.ndarray] = field(default_factory=dict) # side -> mask
    was_fused:       bool = False

# ---------------------------------------------------------------------------
# 2. Factory helpers
# ---------------------------------------------------------------------------

def build_agents_from_segments(
    frame_segments: list[dict],
    depth_min: float = 0.0,
    depth_max: float = 1.0,
) -> list[DetectedAgent]:
    """
    Converts one frame's segment list from SegmentationPipeline.process_vision()
    into DetectedAgent objects, computing distance_weight via global depth
    normalisation (lower depth → closer → higher weight → higher danger).

    Args
    ----
    frame_segments : list of segment dicts for a single frame (all sides).
    depth_min      : global minimum mode_depth across the full batch.
    depth_max      : global maximum mode_depth across the full batch.

    Returns
    -------
    list[DetectedAgent]
    """
    agents      = []
    depth_range = depth_max - depth_min if depth_max > depth_min else 1.0

    # Normalize nested inputs, then merge cross-side masks before creating agents.
    normalized_segments = []
    for seg_entry in frame_segments:
        segs_to_process = seg_entry if isinstance(seg_entry, list) else [seg_entry]
        normalized_segments.extend(seg for seg in segs_to_process if isinstance(seg, dict))

    _SIDE_OVERLAP_MATCHER.merge_side_overlaps(normalized_segments)

    # Group matching segments by instance_id so each physical agent appears once.
    instance_groups: dict[int, list[dict]] = {}
    for seg in normalized_segments:
        iid = seg.get("instance_id", -1)
        if iid not in instance_groups:
            instance_groups[iid] = []
        instance_groups[iid].append(seg)

    agents = []
    for iid, segs in instance_groups.items():
        # Find the closest view for primary side and depth
        best_seg = min(segs, key=lambda x: x.get("mode_depth", 0.5))

        raw_depth = float(best_seg.get("mode_depth", 0.5))
        clamped = max(depth_min, min(depth_max, raw_depth))
        normalized_depth = (clamped - depth_min) / depth_range
        distance_weight = round(1.0 - normalized_depth, 4)

        # Collect all masks and sides
        masks = {s.get("side", "unknown"): s.get("mask") for s in segs}
        fused_sides = list(masks.keys())

        # Create a stable ID based on instance_id
        agent_id = f"agent_{iid}"

        agents.append(DetectedAgent(
            agent_id        = agent_id,
            class_name      = best_seg["class_name"],
            side            = best_seg.get("side", "unknown"),
            frame           = best_seg.get("frame", 0),
            seg_score       = float(best_seg.get("score") or 0.0),
            mode_depth      = raw_depth,
            distance_weight = distance_weight,
            instance_id     = iid,
            fused_sides     = fused_sides,
            masks           = masks,
            was_fused       = len(fused_sides) > 1,
        ))

    return agents


# ? normalization of depth across frames of the epoch
def compute_depth_bounds(segmented_items: list[list[dict]]) -> tuple[float, float]:
    """
    Scans the entire batch of frames to find global (min, max) mode_depth values
    for consistent distance_weight normalisation across all frames.

    Args
    ----
    segmented_items : full output of SegmentationPipeline.process_vision().

    Returns
    -------
    (depth_min, depth_max)
    """
    all_depths = []
    for frame in segmented_items:
        for seg in frame:
            if isinstance(seg, list):
                for s in seg:
                    if isinstance(s, dict) and s.get("mode_depth") is not None:
                        all_depths.append(s["mode_depth"])
            elif isinstance(seg, dict) and seg.get("mode_depth") is not None:
                all_depths.append(seg["mode_depth"])
    if not all_depths:
        return 0.0, 1.0
    return float(min(all_depths)), float(max(all_depths))


# ---------------------------------------------------------------------------
# 3. Prompt builders (pure functions — no API calls)
# ---------------------------------------------------------------------------

def _build_epoch_scene_block(
    env_description: str,
    epoch_agents:    list[list[DetectedAgent]],
    frames:          any,
    telemetry:       Optional[TelemetryData] = None,
    agent_risks:     dict[str, float] = None,
    epoch_id:        str = "latest",
    exclude_ids:     list[str] = None,
    include_multimodal: bool = True,
) -> tuple[str, list[dict]]:
    """
    Assembles a temporal scene context block across a sequence/batch of frames.
    Instead of per-frame listings, it groups agents into trajectories (tracks)
    to ensure continuity and prevent redundant entries for the same road user.
    """
    total_frames = len(epoch_agents)
    exclude_ids = exclude_ids or []

    # 1. Identify all unique physical agents across the entire epoch (by instance_id)
    all_physical_agents: dict[int, DetectedAgent] = {}
    for frame in epoch_agents:
        for a in frame:
            # We store the first occurrence of the physical agent as the representative
            if a.instance_id not in all_physical_agents:
                all_physical_agents[a.instance_id] = a

    # 2. Select the top 10 most critical unique physical agents
    if agent_risks:
        # Aggregate risks by instance_id (max risk across sides for that physical agent)
        instance_risks = {}
        for frame in epoch_agents:
            for a in frame:
                risk = agent_risks.get(a.agent_id, 0.0)
                instance_risks[a.instance_id] = max(instance_risks.get(a.instance_id, 0.0), risk)

        sorted_instances = sorted(
            [iid for iid, a in all_physical_agents.items() if a.agent_id not in exclude_ids],
            key=lambda iid: instance_risks.get(iid, 0.0),
            reverse=True
        )
    else:
        # Fallback: sort by the minimum depth seen across the epoch for that physical agent
        instance_min_depths = {}
        for frame in epoch_agents:
            for a in frame:
                instance_min_depths[a.instance_id] = min(instance_min_depths.get(a.instance_id, float('inf')), a.mode_depth)
        sorted_instances = sorted(
            [iid for iid, a in all_physical_agents.items() if a.agent_id not in exclude_ids],
            key=lambda iid: instance_min_depths[iid]
        )

    top_instance_ids = sorted_instances[:10]

    # 3. Build trajectories for these top physical agents
    tracks: dict[int, list[DetectedAgent]] = {iid: [] for iid in top_instance_ids}
    for frame in epoch_agents:
        for a in frame:
            if a.instance_id in tracks:
                tracks[a.instance_id].append(a)

    # 4. Compute trends and summarize for each track
    track_summaries = []
    multimodal_content = []

    for iid in top_instance_ids:
        occurrences = tracks[iid]
        if not occurrences:
            continue

        first, last = occurrences[0], occurrences[-1]

        # Depth trend and proximity danger
        min_depth = min(a.mode_depth for a in occurrences)
        is_danger_zone = min_depth < CONFIG["risk_assessment"].get("danger_zone_threshold", 0.1)

        if len(occurrences) > 1:
            depth_delta = last.mode_depth - first.mode_depth
            if depth_delta < -0.05:
                trend = "APPROACHING / CLOSING IN (HIGHER DANGER)"
            elif depth_delta > 0.05:
                trend = "RECEDING / MOVING AWAY"
                if is_danger_zone:
                    trend += " (HIGHER DANGER)"
            else:
                trend = "STATIONARY / CONSTANT DISTANCE"
                if is_danger_zone:
                    trend += " (HIGHER DANGER)"
        else:
            trend = "UNKNOWN FLOW"
            if is_danger_zone:
                trend += " (HIGHER DANGER)"

        # Proximity summary
        avg_prox = np.mean([a.distance_weight for a in occurrences]) * 100
        frames_active = f"{occurrences[0].frame}-{occurrences[-1].frame}"

        fused_info = ""
        if first.was_fused and first.fused_sides:
            fused_info = f" [fused: {', '.join(first.fused_sides)}]"

        track_summaries.append(
            f"    • [ID: {first.agent_id}] {first.class_name}{fused_info} | Side: {first.side} | Min Depth: {min(a.mode_depth for a in occurrences):.3f} | "
            f"Frames: {frames_active} | Avg Prox: {avg_prox:.0f}% | Trend: {trend}"
        )

    if include_multimodal:
        # Generate the composite once for the initial WM payload.
        try:
            composite_video_b64 = _generate_epoch_composite_video(top_instance_ids, epoch_agents, frames)
            if composite_video_b64:
                multimodal_content.append({"type": "video", "content": composite_video_b64, "description": "Composite epoch view"})
        except Exception as e:
            logger.warning("Failed to generate composite video, falling back to per-agent videos: %s", e)
            for iid in top_instance_ids:
                rep = all_physical_agents[iid]
                try:
                    video_b64 = _generate_agent_bbox_video(rep.agent_id, epoch_agents, frames)
                    if video_b64:
                        multimodal_content.append({"type": "video", "content": video_b64, "agent_id": rep.agent_id})
                except Exception:
                    pass

    # Build the final text block
    epoch_agents_block = "\n".join(track_summaries) if track_summaries else "  (no dynamic agents detected across epoch)"

    sides_present = sorted(set(a.side for a in [a for f in epoch_agents for a in f]))

    if telemetry is not None:
        telemetry_block = (
            f"  TTC={telemetry.ttc:.2f}s  |  PET={telemetry.pet:.2f}s  |  "
            f"Speed={telemetry.speed_kmh:.1f} km/h  |  v_rel={telemetry.v_rel:.1f} m/s  |  "
            f"d_lat={telemetry.d_lat:.1f} m  |  "
            f"roll={telemetry.roll:.1f}°  |  yaw_rate={telemetry.yaw_rate:.1f} °/s"
        )
    else:
        telemetry_block = "  (not available — IMU/physics sensor integration pending)"

    text = (
        f"[Epoch Temporal Context]\n"
        f"Sliding Window / Epoch Batch: {total_frames} frames  |  Camera sides observed: {', '.join(sides_present) or 'none'}\n\n"
        f"[Environment]\n{env_description}\n\n"
        f"[Agent Trajectories & Temporal Trends across Epoch]\n"
        f"{epoch_agents_block}\n\n"
        f"[Physical Telemetry]\n{telemetry_block}"
    )

    if include_multimodal:
        # Save the initial WM payload for inspection; sampling rebuilds only text.
        try:
            prompt_debug_path = DEBUG_WM_DIR / f"wm_prompt_{epoch_id}.md"
            with open(prompt_debug_path, "w", encoding="utf-8") as f:
                f.write(f"# World Model Input Prompt\n\n## Text Prompt\n\n{text}\n\n## Multimodal Content\n")
                for i, item in enumerate(multimodal_content):
                    if item["type"] == "video":
                        f.write(f"- Video {i}: {item.get('description', 'No description')} (ID: {item.get('agent_id', 'N/A')})\n")
                        video_filename = f"composite_{epoch_id}_{i}.mp4"
                        with open(DEBUG_WM_DIR / video_filename, "wb") as vf:
                            vf.write(base64.b64decode(item["content"]))
                    elif item["type"] == "image":
                        f.write(f"- Image {i}: (ID: {item.get('agent_id', 'N/A')})\n")
                        image_filename = f"crop_{epoch_id}_{i}.jpg"
                        with open(DEBUG_WM_DIR / image_filename, "wb") as ifile:
                            ifile.write(base64.b64decode(item["content"]))
        except Exception as e:
            logger.warning("Failed to save debug prompt: %s", e)

    return text, multimodal_content


def _build_scene_block(
    env_description: str,
    agents:          list[DetectedAgent],
    frames:          any,
    telemetry:       Optional[TelemetryData],
    agent_risks:     dict[str, float] = None,
    exclude_ids:     list[str] = None,
) -> tuple[str, list[dict]]:
    """
    Assembles the shared scene-context block for single-frame evaluation.
    """
    sides_present = sorted(set(a.side for a in agents))
    multimodal_content = []
    exclude_ids = exclude_ids or []
    if agents:
        # 1. Filter excluded agents and then sort by risk (descending) or proximity (closest first)
        filtered_agents = [a for a in agents if a.agent_id not in exclude_ids]
        if agent_risks:
            sorted_agents = sorted(filtered_agents, key=lambda a: agent_risks.get(a.agent_id, 0.0), reverse=True)
        else:
            sorted_agents = sorted(filtered_agents, key=lambda a: a.mode_depth)


        # 2. Limit to 10 dynamic agents
        top_agents = sorted_agents[:10]

        # 3. Group by frame and side
        groups = {}
        for a in top_agents:
            key = (a.frame, a.side)
            if key not in groups:
                groups[key] = []
            groups[key].append(a)

        agent_lines = []
        # 4. Build grouped block
        for (frame, side) in sorted(groups.keys()):
            agent_lines.append(f"  Frame {frame} | Side {side}:")
            for a in groups[(frame, side)]:
                fused_info = ""
                if a.was_fused:
                    sides_str = ", ".join(a.fused_sides) if a.fused_sides else "multiple"
                    fused_info = f" [fused: {sides_str}]"

                agent_lines.append(
                    f"    • [ID: {a.agent_id}] {a.class_name}{fused_info} "
                    f"| Depth: {a.mode_depth:.3f} (Prox: {a.distance_weight * 100:.0f}%) "
                    f"| Conf: {a.seg_score * 100:.0f}%"
                )
                # Extract image for this agent
                try:
                    f_idx = a.frame
                    s = a.side
                    mask = a.masks.get(s)
                    if frames and mask is not None:
                        frame_img = frames[s][f_idx] if isinstance(frames, dict) else frames[f_idx]
                        img_b64 = _generate_agent_bbox_image(a.agent_id, frame_img, mask)
                        if img_b64:
                            multimodal_content.append({"type": "image", "content": img_b64, "agent_id": a.agent_id})
                except Exception:
                    pass

        agents_block = "\n".join(agent_lines)
    else:
        agents_block = "  (no dynamic agents detected)"

    if telemetry is not None:
        telemetry_block = (
            f"  TTC={telemetry.ttc:.2f}s  |  PET={telemetry.pet:.2f}s  |  "
            f"Speed={telemetry.speed_kmh:.1f} km/h  |  v_rel={telemetry.v_rel:.1f} m/s  |  "
            f"d_lat={telemetry.d_lat:.1f} m  |  "
            f"roll={telemetry.roll:.1f}°  |  yaw_rate={telemetry.yaw_rate:.1f} °/s"
        )
    else:
        telemetry_block = "  (not available — IMU/physics sensor integration pending)"

    text = (
        f"[Environment]\n{env_description}\n\n"
        f"[Detected Agents]  (camera sides observed: {', '.join(sides_present) or 'none'})\n"
        f"{agents_block}\n\n"
        f"[Physical Telemetry]\n{telemetry_block}"
    )
    return text, multimodal_content


def _build_reason_prompt(scene_block: str, already_extracted_hazards: list[str] = None) -> str:
    """
    Step 1 prompt — asks the model to produce a concise, actionable dynamic hazard
    in a defensive-driving JSON format.
    """
    criteria_text = "\n".join(
        f"  {i+1}. {name}: {desc}"
        for i, (name, desc) in enumerate(EVALUATION_CRITERIA)
    )

    # Format already extracted hazards as a simple list
    hazards_text = "None"
    if already_extracted_hazards:
        hazards_text = "\n".join([f" - {hid}" for hid in already_extracted_hazards])

    try:
        with open("src/tasks/prompts/reason_generation.md", "r") as f:
            template = f.read()
        return template.format(
            evaluation_criteria=EVALUATION_CRITERIA, # This is a list, but the template expects text?
            already_extracted_hazards=hazards_text,
            criteria_text=criteria_text,
            scene_block=scene_block
        )
    except Exception as e:
        logger.error("Failed to load reason_generation.md: %s", e)
        # Fallback to hardcoded if file is missing
        return (
            "SYSTEM ROLE:\n"
            "You are an expert Defensive Driving AI for a two-wheeled vehicle (motorcycle/bicycle/e-bike). "
            "Your goal is to analyze a 1-second 360-degree video snippet, cross-reference it with the provided "
            "object list and distance metrics, and identify actionable dynamic hazards.\n\n"
            "STRICT DO NOT USE / NEGATIVE CONSTRAINTS:\n"
            "1. NEVER mention camera attributes, field of view, mounting position, or system capabilities "
            "(e.g., DO NOT say 'The rider has a 360 view', 'The camera detects...', 'Because of the lens...').\n"
            "2. NEVER use conversational filler, meta commentary, or chain-of-thought language such as 'Okay, let me think',\n"
            "   'The user wants me to...', 'I see', 'as an AI', or any sentence that is not a hazard observation.\n"
            "3. NEVER describe static environment features as hazards unless they actively restrict trajectory or visibility "
            "(e.g., DO NOT say 'There is a parked car.' SAY 'The parked SUV obstructs visibility of emerging pedestrians from the right sidewalk').\n"
            "4. NEVER state the obvious presence of moving objects without a hazard mechanism "
            "(e.g., DO NOT say 'A car is driving next to me.' SAY 'The sedan on the left is matching speed in my blind spot, blocking lateral evasive maneuvers').\n"
            "5. NEVER output a safe placeholder like 'No additional distinct hazard produced.' unless the scene is truly clear;\n"
            "   if the scene is clear, return an empty hazards array [] inside the JSON object.\n"
            "6. NEVER invent object IDs, distances, or lane geometry that are not supported by the scene data.\n"
            "7. You MUST use the exact `object_id` provided in the [Detected Agents] or [Agent Trajectories] sections. Do not invent new IDs.\n"
            "8. If the hazard relates to the overall environment (e.g., road surface, weather, traffic density), you MUST use `object_id: 'environment'`.\n\n"
            "=== ALREADY EXTRACTED HAZARDS ===\n"
            f"{hazards_text}\n"
            "If this section is not empty, DO NOT produce a hazard for any of the object_ids listed here. Look for DIFFERENT hazards or agents.\n\n"
            "GROUNDING RULES FOR THE HAZARD:\n"
            "- observation: one concrete, physically grounded sentence describing what is visible or moving. Use terms like\n"
            "  'left-side sedan', 'wet metal seam', 'narrowing gap', 'approaching cyclist', or 'braking lead vehicle'.\n"
            "  Mention specific high-risk details: if it is a child, if the agent is facing the road, or if turn signals are active.\n"
            "  IMPORTANT: The ego-vehicle getting cut off is a very high risk situation, especially when the ego-rider intends to turn.\n"
            "- danger_reasoning: explain the mechanism in 2-3 detailed clauses: how it threatens the rider, why it matters now "
            "  (e.g., 'child may dart into road', 'facing road suggests imminent entry'), and what specific action is needed. "
            "  Be explicit about vulnerability and unpredictable intent. Do not mention being an AI or discussing the prompt.\n"
            "- actionable_risk: give imperative defensive-driving guidance such as 'Reduce speed and hold the left edge' or\n"
            "  'Prepare a controlled swerve to the right'.\n"
            "- You must produce EXACTLY ONE most-significant hazard. If no grounded threat exists, return an empty array [].\n\n"
            "EVALUATION FRAMEWORK (Analyze hazards across these 7 categories):\n"
            "1. Trajectory Conflict & Time-to-Collision (TTC):\n"
            "   - Vehicles turning across the path (Dooring, left turns, sudden lane cuts).\n"
            "   - Speed/distance differentials based on the provided object bounding boxes.\n"
            "   - High-risk interactions: bikes and pedestrians in close proximity.\n"
            "2. Vulnerability & Predictability:\n"
            "   - Presence of children (extremely high risk due to unpredictability).\n"
            "   - Road users facing the road (high risk of sudden entry).\n"
            "   - Turn signals of other road users indicating intent to cross or merge.\n"
            "3. Visibility Obscuration & Blind Spots:\n"
            "   - Sightline blockages caused by large vehicles, pillars, or street furniture.\n"
            "   - Areas where a hazard could emerge within < 1.5 seconds.\n"
            "4. Surface & Traction Degradation:\n"
            "   - Road surface hazards specifically dangerous to 2-wheelers (manhole covers, gravel, wet metal, track rails, sudden asphalt changes).\n"
            "5. Ego-Vehicle Trajectory Constraints:\n"
            "   - Escape routes: Is the rider boxed in on the left/right?\n"
            "   - Following distance: Is the vehicle ahead braking or stopping abruptly?\n"
            "6. Rider Awareness & Attention:\n"
            "   - Is the rider likely aware of the hazard? Risk is significantly higher if a threat is closing in from a blind spot, an unexpected angle, or if the rider's current trajectory and speed show no defensive response to an approaching agent.\n"
            "7. Signage, Signals & Lane Context:\n"
            "   - Traffic Signals: Does a red light or stop sign create a high-risk zone for unexpected movements?\n"
            "   - Lane Geometry: Does the rider's position (e.g., riding on the edge) or the lane type (merging, narrowing) elevate risk?\n"
            "   - Markings: Are solid markings being violated or are there restrictive lane boundaries?\n\n",
            "=== SCENE DATA (EPOCH BATCH) ===\n"
            f"{scene_block}\n\n"
            "=== EVALUATION CRITERIA ===\n"
            f"{criteria_text}\n\n"
            "=== OUTPUT FORMAT ===\n"
            "Produce a single JSON object. The output must be valid JSON only. No markdown fences. No commentary. No filler text.\n"
            "Use this structure exactly, containing only the single most important hazard in the hazards array:\n"
            "{\n"
            "  \"hazards\": [\n"
            "    {\n"
            "      \"object_id\": \"<ID_from_input_list_if_applicable>\",\n"
            "      \"hazard_type\": \"<Collision Risk | Visibility Blocker | Surface Hazard | Trajectory Constraint>\",\n"
            "      \"location_relative\": \"<e.g., 2 o'clock, 5 meters ahead>\",\n"
            "      \"observation\": \"<Concrete scene-grounded description of the visible object or road state>\",\n"
            "      \"danger_reasoning\": \"<Short mechanism: why the rider is exposed or can lose control within the next 1-2 seconds>\",\n"
            "      \"actionable_risk\": \"<Defensive-driving instruction>\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Example of a valid hazard object:\n"
            "{\"hazards\":[{\"object_id\":\"car_12\",\"hazard_type\":\"Collision Risk\",\"location_relative\":\"left rear quarter\",\"observation\":\"A sedan is cutting left while closing to the rider's lane edge.\",\"danger_reasoning\":\"The rider's left-side escape route is shrinking and the vehicle is encroaching into the shared path, leaving little time to brake or swerve.\",\"actionable_risk\":\"Reduce speed and keep a wider left buffer.\"}]}\n"
            "Important: if the scene contains no physically grounded threat, return \"hazards\": [] as the array inside the JSON object. Do not fabricate a 'no hazard' narrative.\n"
            "Return only JSON and no extra prose.\n\n"
            "JSON Response:"
        )


def _build_reason_scoring_prompt(scene_block: str, reason: tuple[str, float], reason_index: int) -> str:
    """
    Builds a prompt for the per-reason scoring phase.
    Reasons are generated in one call, but each reason is scored independently
    with its own model request so that different interpretations cannot collapse
    to the same score.
    """
    text, probability = reason
    human = _humanize_hazards(text) if isinstance(text, (str, dict)) else str(text)

    return (
        "You are a master road-safety judge. Your task is to assign a risk score distribution to a specific hazard interpretation in isolation.\n\n"
        "=== SCENE DATA ===\n"
        f"{scene_block}\n\n"
        "=== HAZARD INTERPRETATION TO SCORE ===\n"
        f"Interpretation {reason_index + 1} (Confidence: {probability:.2%}):\n{human}\n\n"
        "=== RIGID RUBRIC (Use this to determine the score) ===\n"
        "1 = MINIMAL RISK  — Clear road; all agents distant; no action required.\n"
        "2 = LOW RISK      — Light traffic nearby; standard defensive riding applies.\n"
        "3 = MODERATE RISK — Elevated caution required; one or more agents are close "
        "or behaving unpredictably; a speed or position adjustment may be needed.\n"
        "4 = HIGH RISK     — Significant hazard; braking or evasive manoeuvre likely "
        "required soon; collision window is opening.\n"
        "5 = SEVERE RISK   — Imminent hazard; emergency manoeuvre required right now "
        "or collision is unavoidable.\n\n"
        "JUDGING RULES:\n"
        "1. Be critical: if the hazard description implies an immediate threat, do not default to 3. Use 4 or 5.\n"
        "2. Be grounded: if the scene data contradicts the interpretation, score it lower.\n"
        "3. Differentiate: avoid assigning a 'neutral' 3 if the evidence strongly points to a specific rubric level.\n"
        "4. Amplify Vulnerability: Scenarios involving children, pedestrians/cyclists in close proximity, or "
        "   unpredictable intent (e.g., facing the road without moving) should strongly push the score toward 4 or 5.\n"
        "5. Amplify High-Risk Maneuvers: Scenarios where the ego-vehicle is being cut off, especially when the "
        "   ego-rider intends to turn, should be treated as very high risk and push the score toward 4 or 5.\n"
        "6. Amplify Awareness & Signal Risk: Scenarios where a hazard is closing in while the rider appears unaware (e.g., from a blind spot), or where critical traffic signals (red lights/stop signs) or lane markings are being ignored, should strongly push the score toward 4 or 5.\n"
        "7. Prioritize Blind-Spot Risks: Hazards involving agents behind the ego-vehicle (Side: back), especially those partially obstructed, should be treated with extreme caution and strongly push the score toward 4 or 5, as they represent a critical lack of visibility for the rider.\n"
        "8. Mapping Logic: A 'Severe Hazard' or 'Imminent Collision' is a Score 5. A 'Clear Road' or 'No Hazard' is a Score 1. Do not invert this logic.\n\n"
        "PROBABILITY DISTRIBUTION RULES:\n"
        "1. The values in `score_probs` must sum exactly to 1.0.\n"
        "2. Expected Value Calculation: The final risk score is computed as a weighted average. For this reason, the model will calculate an 'expected score' = Σ (score * probability). For example, a distribution of {'4': 0.8, '5': 0.2} results in an expected score of 4.2.\n"
        "3. Concentration: If the evidence strongly supports a specific score, assign the vast majority of the probability (e.g., 0.8 to 1.0) to that specific score to create a strong, clear signal.\n"
        "4. Avoid 'Flat' Distributions: Do not assign minimal equal probabilities (e.g., 0.2 across all) unless you are genuinely uncertain. A flat distribution always results in an expected score of 3.0, which dilutes the risk signal and may mask severe hazards or overstate minimal risks.\n\n"
        "RESPONSE FORMAT:\n"
        "Produce exactly one JSON object. Do not emit any other text, score digits, or commentary outside the JSON.\n"
        "The JSON must follow this schema:\n"
        "{\n"
        "  \"reason_index\": <int>,\n"
        "  \"score_probs\": {\n"
        "    \"1\": <float>,\n"
        "    \"2\": <float>,\n"
        "    \"3\": <float>,\n"
        "    \"4\": <float>,\n"
        "    \"5\": <float>\n"
        "  },\n"
        "  \"critique\": \"one-sentence grounding explaining the distribution and why the peak score was chosen over adjacent levels\"\n"
        "}\n"
        "Note: The probabilities in `score_probs` must sum to 1.0.\n\n"
        "JSON Response:"
    )


# ---------------------------------------------------------------------------
# 4. Helper utilities
# ---------------------------------------------------------------------------

VIDEO_CACHE_DIR = Path("outputs/risk_videos")

def get_color_for_agent(agent_id: str) -> tuple[int, int, int]:
    """
    Returns a stable BGR color based on the agent_id.
    """
    import zlib
    palette = [
        (0, 255, 0),   # Green
        (0, 0, 255),   # Red
        (255, 0, 0),   # Blue
        (0, 255, 255), # Yellow
        (255, 0, 255), # Magenta
        (255, 255, 0), # Cyan
        (0, 165, 255), # Orange
        (128, 0, 128), # Purple
        (255, 165, 0), # Blue-Orange
        (0, 128, 128), # Teal
    ]
    idx = zlib.adler32(agent_id.encode()) % len(palette)
    return palette[idx]


def _correct_object_id(returned_id: str, valid_ids: list[str], id_to_class: dict[str, str] = None) -> str:
    """
    Ensures the returned object_id is grounded in the provided valid_ids.
    If not exactly matching or 'environment', it tries to find the closest match
    based on numeric IDs or class keywords.
    """
    if not returned_id or returned_id == "environment":
        return returned_id

    if returned_id in valid_ids:
        return returned_id

    # 1. Attempt to find a match based on numeric ID
    import re
    nums = re.findall(r'\d+', returned_id)
    if nums:
        primary_num = nums[0]
        for vid in valid_ids:
            if primary_num in vid:
                return vid

    # 2. Attempt to find a match based on class keywords (if mapping provided)
    if id_to_class:
        returned_id_low = returned_id.lower()
        for vid, class_name in id_to_class.items():
            if class_name.lower() in returned_id_low:
                return vid

    # Fallback: return "unknown" if no match found
    return "unknown"

def _generate_agent_bbox_image(
    agent_id: str,
    frame: np.ndarray,
    mask: np.ndarray
) -> str:
    """
    Generates an image for a specific agent in a single frame.
    Draws a bounding box and ID around the agent.
    Returns a base64 JPEG.
    """
    if mask is None or not np.any(mask):
        return ""

    img = frame.copy()
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    color = get_color_for_agent(agent_id)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)

    # Label with background
    text = agent_id
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_x, text_y = x0, max(0, y0 - 10)
    cv2.rectangle(img, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline), color, -1)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode("utf-8")

def _generate_epoch_composite_video(
    top_instance_ids: list[int],
    epoch_agents: list[list[DetectedAgent]],
    frames: any
) -> str:
    """
    Generates a single composite video for an epoch, stitching all camera sides
    horizontally and drawing bounding boxes for the top physical agents.
    Caches the video and returns the base64 encoded video string.
    """
    VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Define the desired order of sides for the composite strip
    target_sides = ['left', 'front', 'right', 'back']
    # Only use sides that are actually present in the frames data
    available_sides = []
    if isinstance(frames, dict):
        available_sides = [s for s in target_sides if s in frames]
    elif isinstance(frames, list):
        available_sides = ['front'] # Assume list is front if not specified

    if not available_sides:
        return ""

    overlap = max(0.0, float(CONFIG.get("processing", {}).get("side_overlap", 0.0)))
    cache_digest = hashlib.sha256()
    cache_digest.update(str((overlap, sorted(top_instance_ids))).encode())
    for side in available_sides:
        side_frames = frames[side] if isinstance(frames, dict) else frames
        cache_digest.update(side.encode())
        for image in side_frames:
            cache_digest.update(np.ascontiguousarray(image).view(np.uint8))
    for frame_idx, frame in enumerate(epoch_agents):
        for agent in frame:
            if agent.instance_id not in top_instance_ids:
                continue
            cache_digest.update(f"{frame_idx}:{agent.instance_id}".encode())
            for side, mask in sorted(agent.masks.items()):
                if mask is not None:
                    cache_digest.update(side.encode())
                    cache_digest.update(np.ascontiguousarray(mask, dtype=np.uint8).view(np.uint8))

    cache_key = "_".join(map(str, sorted(top_instance_ids)))[:80] or "no_agents"
    cache_key = f"{cache_key}_{cache_digest.hexdigest()[:16]}"
    video_path = VIDEO_CACHE_DIR / f"composite_{cache_key}.mp4"
    num_frames = len(epoch_agents)
    if video_path.exists():
        cached_video = cv2.VideoCapture(str(video_path))
        cached_frame_count = int(cached_video.get(cv2.CAP_PROP_FRAME_COUNT)) if cached_video.isOpened() else 0
        cached_video.release()
        if cached_frame_count == num_frames:
            with open(video_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        video_path.unlink(missing_ok=True)

    video_frames = []
    # Iterate through the epoch frames
    for f_idx in range(num_frames):
        # 1. Build the composite frame by stitching available sides
        side_imgs = []
        side_offsets = {}
        side_margins = {}
        current_offset = 0
        for side in available_sides:
            img = frames[side][f_idx] if isinstance(frames, dict) else frames[f_idx]
            margin = int((overlap / (2.0 + 2.0 * overlap)) * img.shape[1]) if overlap else 0
            cropped_img = img[:, margin:img.shape[1] - margin] if margin else img
            side_offsets[side] = current_offset
            side_margins[side] = margin
            side_imgs.append(cropped_img)
            current_offset += cropped_img.shape[1]

        composite_img = np.hstack(side_imgs).copy()

        # 2. Draw BBoxes for all top agents present in this frame
        placed_labels = []  # Track (x, y, w, h) of labels in this frame
        for agent in epoch_agents[f_idx]:
            if agent.instance_id not in top_instance_ids:
                continue

            composite_mask = np.zeros(composite_img.shape[:2], dtype=bool)
            for side in agent.fused_sides:
                if side not in side_offsets:
                    continue
                mask = agent.masks.get(side)
                if mask is None or not np.any(mask):
                    continue

                margin = side_margins[side]
                cropped_mask = mask[:, margin:mask.shape[1] - margin] if margin else mask
                side_start = side_offsets[side]
                side_end = min(side_start + cropped_mask.shape[1], composite_img.shape[1])
                if side_end > side_start:
                    composite_mask[:, side_start:side_end] |= cropped_mask[:, :side_end - side_start]

            if not np.any(composite_mask):
                continue

            color = get_color_for_agent(agent.agent_id)
            overlay = composite_img.copy()
            overlay[composite_mask] = color
            cv2.addWeighted(overlay, 0.5, composite_img, 0.5, 0, composite_img)

            coords = np.argwhere(composite_mask)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            cv2.rectangle(composite_img, (x0, y0), (x1, y1), color, 3)

            text = f"ID:{agent.agent_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            label_x = int(x0)
            label_y = max(0, int(y0) - 10)

            for shift in range(0, 100, 20):
                label_rect = (label_x, label_y - th - baseline - shift, label_x + tw, label_y + baseline - shift)
                overlaps_label = any(
                    not (label_rect[2] <= prev[0] or label_rect[0] >= prev[2]
                         or label_rect[3] <= prev[1] or label_rect[1] >= prev[3])
                    for prev in placed_labels
                )
                if not overlaps_label:
                    label_y -= shift
                    break

            cv2.rectangle(composite_img, (label_x, label_y - th - baseline), (label_x + tw, label_y + baseline), color, -1)
            cv2.putText(composite_img, text, (label_x, label_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            placed_labels.append((label_x, label_y - th - baseline, label_x + tw, label_y + baseline))

        video_frames.append(composite_img)

    if not video_frames:
        return ""

    # Save as video
    h, w = video_frames[0].shape[:2]
    output_size = (w - w % 2, h - h % 2)
    if output_size[0] < 2 or output_size[1] < 2:
        raise ValueError(f"Invalid WM composite video dimensions: {w}x{h}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, output_size)
    if not out.isOpened():
        out.release()
        video_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not open WM composite video writer for {video_path} at {output_size}")
    try:
        for frame_idx, frame in enumerate(video_frames):
            if frame.shape[:2] != (output_size[1], output_size[0]):
                logger.warning(
                    "Resizing WM composite frame %d from %sx%s to %sx%s",
                    frame_idx,
                    frame.shape[1],
                    frame.shape[0],
                    output_size[0],
                    output_size[1],
                )
                frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
    finally:
        out.release()

    written_video = cv2.VideoCapture(str(video_path))
    written_frame_count = int(written_video.get(cv2.CAP_PROP_FRAME_COUNT)) if written_video.isOpened() else 0
    written_video.release()
    if written_frame_count != num_frames:
        video_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"WM composite video is incomplete: wrote {written_frame_count} of {num_frames} frames"
        )

    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _generate_agent_bbox_video(
    agent_id: str,
    epoch_agents: list[list[DetectedAgent]],
    frames: any
) -> str:
    """
    Generates a video for a specific agent across an epoch.
    Draws a bounding box and ID on each frame where the agent is present.
    Caches the video and returns the base64 encoded video string.
    """
    VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    video_path = VIDEO_CACHE_DIR / f"{agent_id}.mp4"

    if video_path.exists():
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Find all appearances of the agent to determine the primary side
    agent_appearances = []
    for f_idx, frame_agents in enumerate(epoch_agents):
        for a in frame_agents:
            if a.agent_id == agent_id:
                agent_appearances.append((f_idx, a))
                break

    if not agent_appearances:
        return ""

    # Use the side of the first appearance as the primary side for the video
    primary_side = agent_appearances[0][1].side
    side_frames = frames[primary_side] if isinstance(frames, dict) else frames

    if not side_frames:
        return ""

    video_frames = []
    for f_idx in range(len(side_frames)):
        img = side_frames[f_idx].copy()

        # Find the agent in this frame
        agent_in_frame = next((a for a in epoch_agents[f_idx] if a.agent_id == agent_id), None)

        if agent_in_frame and agent_in_frame.side == primary_side:
            mask = agent_in_frame.masks.get(primary_side)
            if mask is not None and np.any(mask):
                coords = np.argwhere(mask)
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0) + 1

                # Draw BBox
                color = get_color_for_agent(agent_id)
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)

                # Draw ID with background
                text = agent_id
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_x, text_y = x0, max(0, y0 - 10)
                cv2.rectangle(img, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline), color, -1)
                cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        video_frames.append(img)

    if not video_frames:
        return ""

    # Save as video
    h, w, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (w, h))
    for frame in video_frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()

    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _infer_score_from_text(text: str, score_scale: Optional[list[int]] = None) -> Optional[int]:
    """Infer a likely score from free-form model output when the model does not emit a clean digit."""
    if not text:
        return None

    normalized = text.lower().strip()
    if not normalized:
        return None

    # Prefer explicit digits first, including patterns like "score: 4" or "4/5".
    digit_match = re.search(r"\b([1-5])\b", normalized)
    if digit_match:
        score = int(digit_match.group(1))
        if score_scale is None or score in score_scale:
            return score

    # Fall back to simple keyword cues from the rubric wording.
    keyword_scores = {
        "minimal": 1,
        "clear": 1,
        "low": 2,
        "light": 2,
        "mild": 2,
        "moderate": 3,
        "elevated": 3,
        "noticeable": 3,
        "high": 4,
        "significant": 4,
        "severe": 5,
        "imminent": 5,
        "urgent": 5,
        "emergency": 5,
        "critical": 5,
    }
    for keyword, score in keyword_scores.items():
        if keyword in normalized:
            if score_scale is None or score in score_scale:
                return score

    return None


def _canonical_reason_payload(reason: dict) -> str:
    """Build a stable signature for deduplicating hazard payloads."""
    hazards = reason.get("hazards") if isinstance(reason, dict) else []
    if not isinstance(hazards, list):
        return ""
    return json.dumps(hazards, sort_keys=True, ensure_ascii=False)


def _deduplicate_reason_payloads(reason_payloads: list[dict], target_count: int) -> list[dict]:
    """Drop repeated hazard payloads and avoid filling missing slots with invented reasons."""
    unique: list[dict] = []
    seen: set[str] = set()

    for reason in reason_payloads:
        canonical = _canonical_reason_payload(reason)
        if not canonical:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        unique.append(reason)

    # Never invent missing reasons. If the model produced fewer valid candidates,
    # return only the grounded candidates we actually received.
    return unique[:target_count]


def _humanize_hazards(hazard_payload: Union[str, dict]) -> str:
    """Turn a hazards JSON (string or dict) into a concise human-readable block.

    This helps the synthesis judge see the concrete observations and avoid
    scoring based on irrelevant prompt tokens.
    """
    try:
        if isinstance(hazard_payload, str):
            parsed = json.loads(hazard_payload)
        else:
            parsed = hazard_payload
    except Exception:
        return str(hazard_payload)

    lines: list[str] = []
    if isinstance(parsed, list):
        hazards = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("hazards"), list):
        hazards = parsed.get("hazards", [])
    else:
        return str(hazard_payload)

    for i, h in enumerate(hazards):
        oid = h.get("object_id", "unknown")
        htype = h.get("hazard_type", "unknown")
        loc = h.get("location_relative", "unknown")
        obs = h.get("observation", "(no observation)")
        dr = h.get("danger_reasoning", "(no reasoning)")
        blind = "YES" if h.get("is_blind_spot") else "NO"
        obstr = h.get("obstruction_level", "unknown")
        dist = h.get("estimated_distance", "unknown")
        lines.append(f"Hazard {i+1}: [{htype}] {obs} (obj:{oid}, loc:{loc}, dist:{dist})")
        lines.append(f"  Blind Spot: {blind} | Obstruction: {obstr}")
        lines.append(f"  Mechanism: {dr}")

    return "\n".join(lines)


def _strip_meta_preface(text: str) -> str:
    """Remove leading conversational/meta prefacing from model output.

    This strips sentences that look like chain-of-thought or prompt restatement
    (e.g. "Okay, let's break this down...", "As an AI...") and returns the
    remainder starting from the first sentence that contains grounding cues.
    """
    if not text:
        return text

    # Sentence-split conservatively
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 1:
        return text.strip()

    meta_prefix_re = re.compile(r"^(?:ok(?:ay)?|let'?s|i\b|as an ai|the user|dear|note)\b", re.IGNORECASE)
    grounding_keywords = [
        "left", "right", "front", "rear", "pedestrian", "cyclist", "bicycle",
        "vehicle", "car", "truck", "brake", "approach", "approaching", "ttc",
        "proximity", "meter", "m", "speed", "closing", "encroach", "obstruct",
        "visibility", "lane", "collision", "skid", "slide",
    ]

    start_idx = 0
    for i, sent in enumerate(sentences):
        s = sent.strip()
        if not s:
            start_idx = i + 1
            continue

        # If sentence looks explicitly meta/conversational, skip it
        if meta_prefix_re.search(s):
            start_idx = i + 1
            continue

        # If sentence contains any grounding keyword, keep from here
        low = s.lower()
        if any(k in low for k in grounding_keywords):
            start_idx = i
            break

        # otherwise treat short non-grounding sentences as meta and skip
        if len(s.split()) < 6:
            start_idx = i + 1
            continue

        # Default: first reasonably long sentence is probably grounded
        start_idx = i
        break

    remainder = " ".join(sentences[start_idx:]).strip()
    return remainder or text.strip()


def _strip_thinking_blocks(text: str) -> str:
    """Removes <think>...</think> blocks from model output, including unterminated ones."""
    if not text:
        return ""
    # Remove closed blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove unterminated blocks (from <think> to end of string)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def _crop_agent_image(frame: np.ndarray, mask: np.ndarray, size: int = 128) -> str:
    """
    Crops an agent from a frame using its binary mask, resizes it, and returns a base64 JPEG.
    """
    if mask is None or not np.any(mask):
        return ""

    # Get bounding box of the mask
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Crop
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return ""

    # Resize to square
    h, w = crop.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Padding to make it exactly size x size
    pad_h = size - new_h
    pad_w = size - new_w
    resized = cv2.copyMakeBorder(
        resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Encode to base64 JPEG
    _, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode("utf-8")


def _extract_model_content(response) -> str:
    """Return the cleaned model completion text if available, otherwise an empty string."""
    try:
        choice = response.choices[0]
        content = getattr(choice.message, "content", "") or str(choice.message.content)
        return _strip_thinking_blocks(content)
    except Exception:
        return ""


def _log_world_model_call(label: str, prompt: str, response) -> None:
    """Log the full request summary and the raw model output for each world-model call."""
    try:
        msg_p = f"\n{'='*80}\n[PROMPT: {label}]\n{prompt}\n{'='*80}"
        logger.info(msg_p)
        print(msg_p)
    except Exception:
        pass

    try:
        usage = getattr(response, "usage", None)
        if usage:
            print(f"[{label}] Tokens - Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}, Total: {usage.total_tokens}")
    except Exception:
        pass

    try:
        content = _extract_model_content(response)
        msg_o = "[%s] output (trunc): %s" % (label, content[:4000])
        logger.info(msg_o)
        print(msg_o)
    except Exception:
        pass

    _log_model_response(label, response)


def _log_model_response(label: str, response) -> None:
    """Safely log a model response summary for debugging.

    Avoid dumping full objects or secrets; log truncated content, token counts,
    and any usage metadata if available.
    """
    content = _extract_model_content(response)

    try:
        logger.debug("[%s] model content (trunc): %s", label, content[:2000])
    except Exception:
        pass

    try:
        choice = response.choices[0]
        lp_tokens = list(getattr(choice.logprobs, "content", []) or [])
        logger.debug("[%s] token_logprob_count=%d", label, len(lp_tokens))
    except Exception:
        pass

    try:
        usage = getattr(response, "usage", None)
        if usage:
            logger.debug("[%s] usage: %s", label, str(usage))
    except Exception:
        pass


def generate_hazard_viz_video(
    output_path: str,
    frames: any,
    epoch_agents: list[list[DetectedAgent]],
    reasons_detail: list[dict],
    epoch_telemetry: Optional[list[TelemetryData]] = None,
) -> str:
    """
    Generates a video visualization of the top 4 hazards identified in the risk assessment.
    Each hazard is highlighted by its segmentation mask with a distinct color.
    Now supports composite 360° view and correct color spacing, plus telemetry overlay.
    """
    # 1. Identify the top 4 unique hazard IDs based on softmax weight
    valid_reasons = [r for r in reasons_detail if r.get("hazards")]
    sorted_reasons = sorted(
        valid_reasons,
        key=lambda r: r.get("softmax_weight", r.get("weight", 0.0)),
        reverse=True
    )

    top_hazard_ids = []
    for r in sorted_reasons:
        if not r["hazards"]: continue
        oid = r["hazards"][0].get("object_id")
        if oid and oid not in top_hazard_ids:
            top_hazard_ids.append(oid)
        if len(top_hazard_ids) >= 4:
            break

    COLORS = [
        (0, 0, 255),   # Red (in BGR: Blue=0, Green=0, Red=255)
        (0, 165, 255), # Orange
        (0, 255, 255), # Yellow
        (255, 0, 255), # Magenta
    ]

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine available sides for composite view
    target_sides = ['left', 'front', 'right', 'back']
    available_sides = []
    if isinstance(frames, dict):
        available_sides = [s for s in target_sides if s in frames]
    elif isinstance(frames, list):
        available_sides = ['front']

    if not available_sides:
        return ""

    # Setup Video Writer based on the composite width
    # We need to determine the width of one frame first and handle overlap correction
    sample_frame = frames[available_sides[0]][0] if isinstance(frames, dict) else frames[0]
    h, w_single, _ = sample_frame.shape

    # Overlap correction: if side_overlap is configured, crop the margins
    # The projected faces were sampled in range [-1-s, 1+s] and resized to w_single.
    # The center 90-deg region is the middle 1/(1+s) fraction of the image.
    import src.tasks.config.utils
    overlap = src.tasks.config.utils.CONFIG["processing"].get("side_overlap", 0.0)

    if overlap > 0:
        # Calculate the pixel width of the overlap margin on each side
        # The original FOV (90 deg) is 2 units wide. The sampled FOV is 2 + 2*overlap units wide.
        # Margin on one side is 'overlap' units.
        # Margin in pixels: (overlap / (2 + 2*overlap)) * w_single
        margin = int((overlap / (2.0 + 2.0 * overlap)) * w_single)
        effective_w = w_single - 2 * margin
    else:
        margin = 0
        effective_w = w_single

    total_w = effective_w * len(available_sides)

    telemetry_bar_h = 100 if epoch_telemetry else 0
    total_h = h + telemetry_bar_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = Path(output_path)
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (total_w, total_h))

    # Telemetry plot configuration
    metrics = [
        {"name": "Speed (km/h)", "key": "speed_kmh", "color": (0, 255, 0), "min": 0, "max": 120},
        {"name": "Roll (deg)", "key": "roll", "color": (255, 0, 0), "min": -30, "max": 30},
        {"name": "Yaw Rate (deg/s)", "key": "yaw_rate", "color": (0, 255, 255), "min": -60, "max": 60},
        {"name": "Accel (m/s²)", "key": "a_ego", "color": (255, 0, 255), "min": -5, "max": 5},
    ]


    for f_idx in range(len(epoch_agents)):
        # 1. Build the composite frame
        side_imgs = []
        side_offsets = {}
        current_offset = 0
        for side in available_sides:
            img = frames[side][f_idx] if isinstance(frames, dict) else frames[f_idx]

            # Correct overlap: crop the margins from both sides
            if overlap > 0:
                img = img[:, margin : w_single - margin]

            side_imgs.append(img)
            side_offsets[side] = current_offset
            current_offset += img.shape[1]

        composite_img = np.hstack(side_imgs).copy()
        frame_agents = epoch_agents[f_idx]

        # Draw Telemetry Bar
        if epoch_telemetry:
            bar = np.zeros((telemetry_bar_h, total_w, 3), dtype=np.uint8)
            num_metrics = len(metrics)
            slot_w = total_w // num_metrics
            for i, m in enumerate(metrics):
                x_start = i * slot_w
                x_end = (i + 1) * slot_w
                # Draw slot background
                cv2.rectangle(bar, (x_start, 0), (x_end, telemetry_bar_h), (40, 40, 40), -1)
                cv2.putText(bar, m["name"], (x_start + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

                # Draw sparkline
                points = []
                telemetry_samples = min(f_idx + 1, len(epoch_telemetry))
                for f_idx_plot in range(telemetry_samples):
                    val = getattr(epoch_telemetry[f_idx_plot], m["key"], 0.0)
                    norm_val = (val - m["min"]) / (m["max"] - m["min"])
                    y = telemetry_bar_h - 10 - (norm_val * (telemetry_bar_h - 40))
                    x = x_start + 10 + (f_idx_plot / (len(epoch_agents) - 1) * (slot_w - 20)) if len(epoch_agents) > 1 else x_start + 10
                    points.append((int(x), int(y)))

                if len(points) >= 2:
                    cv2.polylines(bar, [np.array(points, np.int32)], False, m["color"], 2, cv2.LINE_AA)

            composite_img = np.vstack([bar, composite_img])


        # 2. Draw Highlights for Top Hazards
        for rank, oid in enumerate(top_hazard_ids):
            color = COLORS[rank] if rank < len(COLORS) else (255, 255, 255)
            agent = next((a for a in frame_agents if a.agent_id == oid), None)

            if agent:
                composite_mask = np.zeros((h, total_w), dtype=bool)
                for side in agent.fused_sides:
                    if side not in side_offsets:
                        continue
                    mask = agent.masks.get(side)
                    if mask is None or not np.any(mask):
                        continue
                    cropped_mask = mask[:, margin:mask.shape[1] - margin] if margin else mask
                    offset_x = side_offsets[side]
                    end_x = min(offset_x + cropped_mask.shape[1], total_w)
                    if end_x > offset_x:
                        composite_mask[:, offset_x:end_x] |= cropped_mask[:, :end_x - offset_x]

                if np.any(composite_mask):
                    full_mask = np.zeros((total_h, total_w), dtype=bool)
                    full_mask[telemetry_bar_h:telemetry_bar_h + h] = composite_mask
                    overlay = composite_img.copy()
                    overlay[full_mask] = color
                    cv2.addWeighted(overlay, 0.5, composite_img, 0.5, 0, composite_img)

                    coords = np.argwhere(composite_mask)
                    y0, x0 = coords.min(axis=0)
                    y1, x1 = coords.max(axis=0) + 1
                    y0 += telemetry_bar_h
                    y1 += telemetry_bar_h
                    cv2.rectangle(composite_img, (x0, y0), (x1, y1), color, 3)
                    label = f"Hazard {rank+1}: {agent.class_name}"

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    text_x, text_y = x0, max(telemetry_bar_h, y0 - 10)
                    cv2.rectangle(composite_img, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline), color, -1)
                    cv2.putText(composite_img, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # 3. Correct Color Format: RGB -> BGR for OpenCV VideoWriter
        final_frame = cv2.cvtColor(composite_img, cv2.COLOR_RGB2BGR)
        out.write(final_frame)

    out.release()

    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# 4. Risk Assessment Engine
# ---------------------------------------------------------------------------

# Constant for environmental hazard distance weight
ENVIRONMENT_DISTANCE_WEIGHT = 0.5

class RiskAssessmentEngine:
    """
    Hybrid risk assessment engine for the 360-degree motorcycle safety pipeline.

    This engine combines the G-VEval mathematical framework with an intelligent
    synthesis judge.

    Pipeline:
    1. Reason Sampling: Generate N independent risk interpretations (r_i).
       Compute probability P(r_i) via softmax-normalized mean log-probabilities.
    2. Smart Scoring: Use a single synthesis call to judge each reason and
       assign a grounded score s_i based on physical evidence.
    3. Mathematical Fusion: The final risk score is the expectation over reasons:
       Final Score = Σ [P(r_i) * s_i]

    This produces a continuous risk value that represents the probability-weighted
    average of the most grounded physical interpretations of the scene.
    """

    def __init__(
        self,
        api_base:   str = _DEFAULT_API_BASE,
        model_name: str = _DEFAULT_MODEL,
        n_reasons:  int = _DEFAULT_N,
    ):
        self.client     = OpenAI(base_url=api_base, api_key="vllm-local")
        self.model_name = model_name
        self.n_reasons  = n_reasons
        self.risk_map_engine = RiskMapEngine()

    def _calculate_agent_risk_values(
        self,
        agents: list[DetectedAgent],
        image: np.ndarray,
        depth_map: np.ndarray,
        side: str,
    ) -> tuple[dict[str, float], dict[str, Hazard]]:
        """
        Uses the RiskMapEngine to compute a risk score for each agent based on the spatial risk map.
        Returns a mapping of agent_id -> mean_risk and agent_id -> matched_hazard.
        """
        # We use the agents' existing masks to generate a risk map that reflects the current scene
        hazards_data = []
        for a in agents:
            mask = a.masks.get(side)
            if mask is None or not np.any(mask):
                continue
            hazards_data.append({
                "id": a.agent_id,
                "mask": mask,
                "score": a.seg_score,
                "conf": a.seg_score,
                "description": f"{a.class_name} agent"
            })

        if not hazards_data:
            return {}, {}

        risk_map, computed_hazards = self.risk_map_engine.generate_risk_map(image, depth_map, hazards_data)

        agent_risks = {}
        agent_hazards = {}

        for a in agents:
            mask = a.masks.get(side)
            if mask is None or not np.any(mask):
                agent_risks[a.agent_id] = 0.0
                continue

            # Calculate mean risk value within the agent's mask
            risk_vals = risk_map[mask]
            agent_risks[a.agent_id] = float(np.mean(risk_vals))

            # Match to the computed Hazard object via ID (since we used agent_id in hazards_data)
            for h in computed_hazards:
                if h.id == a.agent_id:
                    agent_hazards[a.agent_id] = h
                    break

        return agent_risks, agent_hazards

    def _get_distance_weight(
        self,
        object_id: str,
        agents: list[DetectedAgent],
        agent_hazards: dict[str, Hazard] = None,
    ) -> float:
        """Retrieves distance weight for an object; uses semantic risk if a matched Hazard exists."""
        if not object_id or object_id == "unknown":
            return ENVIRONMENT_DISTANCE_WEIGHT

        if object_id == "environment":
            return ENVIRONMENT_DISTANCE_WEIGHT

        # If we have a matched Hazard from RiskMapEngine, use its computed risk components.
        if agent_hazards and object_id in agent_hazards:
            hazard = agent_hazards[object_id]
            weight = max(hazard.c_vlm, 0.2)
            return max(0.0, min(1.0, weight))

        for a in agents:
            if a.agent_id == object_id:
                return a.distance_weight

        return ENVIRONMENT_DISTANCE_WEIGHT

    def _call_llm_with_retry(
        self,
        prompt: str,
        temperature: float,
        multimodal_items: Optional[list[dict]] = None,
        max_tokens: int = 16384,
        logprobs: bool = True,
        top_logprobs: Optional[int] = None,
    ) -> tuple[any, Optional[dict]]:
        """
        Calls the LLM with retries if a valid JSON block is not found in the response.
        Returns a tuple of (response_object, parsed_json_data).
        """
        last_response = None
        for attempt in range(1, 6):
            try:
                # Build multimodal content
                content = [{"type": "text", "text": prompt}]
                if multimodal_items:
                    for item in multimodal_items:
                        if item["type"] == "video":
                            content.append({
                                "type": "video_url",
                                "video_url": {"url": f"data:video/mp4;base64,{item['content']}"}
                            })
                        else:
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{item['content']}"}
                            })

                response = self.client.chat.completions.create(
                    model       = self.model_name,
                    messages    = [{"role": "user", "content": content}],
                    temperature = temperature,
                    max_tokens  = max_tokens,
                    logprobs    = logprobs,
                    top_logprobs = top_logprobs,
                )
                last_response = response

                # Check for truncation
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    logger.warning("Attempt %d/%d: LLM response truncated (finish_reason='length'). Thinking may be too long for max_tokens=%d",
                                   attempt, 5, max_tokens)


                # Extract and clean content
                content_str = _extract_model_content(response)

                # Try to find and parse JSON
                json_match = re.search(r"\{.*\}", content_str, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                        if isinstance(data, dict):
                            # Validate that hazards is a list of dicts if it exists
                            hazards = data.get("hazards")
                            if hazards is not None:
                                if isinstance(hazards, list) and all(isinstance(h, dict) for h in hazards):
                                    return response, data
                                else:
                                    logger.warning("Attempt %d/%d: 'hazards' key found but is not a list of dicts.", attempt, 5)
                            else:
                                # If 'hazards' is missing, we still consider the dict valid JSON
                                return response, data
                    except json.JSONDecodeError:
                        pass

                logger.warning(
                    "Attempt %d/%d: No valid JSON found in LLM response for prompt: %s...",
                    attempt, 5, prompt[:100]
                )
            except Exception as e:
                logger.warning("Attempt %d/%d: LLM call failed with error: %s", attempt, 5, e)

        logger.error("All 5 attempts failed to produce valid JSON for prompt: %s", prompt[:100])
        return last_response, None

    # ------------------------------------------------------------------
    # Step 1 — Reason sampling with logprob weights
    # ------------------------------------------------------------------
    def _generate_reasons_with_logprobs(
        self,
        context: dict,
        multimodal_items:   list[dict] = None,
        valid_ids:          list[str] = None,
        id_to_class:        dict[str, str] = None,
    ) -> list[tuple[str, float]]:
        """
        Calls the model N times independently (temperature > 0) to obtain N
        diverse risk interpretations of the scene.

        Calculates the weight P(r_i) for each reason using the softmax-normalized
        mean per-token log-probabilities.
        """
        reasons_data = []
        mean_lps = []
        extracted_ids = []

        for i in range(self.n_reasons):
            # Rebuild the scene block excluding already extracted IDs to force diversity
            if context.get("is_epoch", False):
                scene_block, current_multimodal = _build_epoch_scene_block(
                    env_description = context["env_description"],
                    epoch_agents    = context["epoch_agents"],
                    frames          = context["frames"],
                    telemetry       = context["telemetry"],
                    agent_risks     = context.get("agent_risks"),
                    epoch_id        = context.get("epoch_id", "latest"),
                    exclude_ids     = extracted_ids,
                    include_multimodal = False,
                )
            else:
                scene_block, current_multimodal = _build_scene_block(
                    env_description = context["env_description"],
                    agents          = context["agents"],
                    frames          = context["frames"],
                    telemetry       = context["telemetry"],
                    agent_risks     = context.get("agent_risks"),
                    exclude_ids     = extracted_ids
                )

            # Use the filtered multimodal content for this specific sample
            active_multimodal = []
            if multimodal_items:
                for item in multimodal_items:
                    if item.get("agent_id") and item.get("agent_id") in extracted_ids:
                        continue
                    active_multimodal.append(item)
            else:
                active_multimodal = current_multimodal

            prompt = _build_reason_prompt(scene_block, already_extracted_hazards=extracted_ids)

            # Use the retry helper for LLM call and JSON extraction
            response, reason_item = self._call_llm_with_retry(
                prompt = prompt,
                temperature = 0.9,
                multimodal_items = active_multimodal,
                logprobs = True,
            )


            if response is None:
                continue

            _log_world_model_call(f"reason_sampling_{i}", prompt, response)

            choice = response.choices[0]
            # If reason_item was None, we apply the existing free-form text fallback
            if reason_item is None:
                raw_content = choice.message.content.strip()
                predicted_text = _strip_thinking_blocks(raw_content)
                text = _strip_meta_preface(predicted_text)
                if text:
                    reason_item = {
                        "hazards": [{
                            "object_id": "error",
                            "hazard_type": "Collision Risk",
                            "location_relative": "unknown",
                            "observation": text[:200],
                            "danger_reasoning": "Model produced free-form text instead of JSON.",
                            "actionable_risk": "Maintain defensive speed.",
                        }]
                    }
                else:
                    # Total failure for this sample
                    continue


            # Sanitize hazards
            for h in reason_item.get("hazards", []):
                if not isinstance(h, dict):
                    continue

                # --- ID Correction Logic ---
                if valid_ids is not None:
                    orig_id = h.get("object_id")
                    corrected_id = _correct_object_id(orig_id, valid_ids, id_to_class=id_to_class)
                    h["object_id"] = corrected_id
                # We can't easily get the list here without passing it in,
                # but we can use the representative list if we modify the function signature
                # or we can do a partial correction if we had the IDs.
                # Since this is a method of RiskAssessmentEngine, we might need to pass the valid IDs.
                # Let's first modify the function signature to accept valid_ids.

                if isinstance(h.get("observation"), str):
                    h["observation"] = _strip_meta_preface(h.get("observation", "")).strip()
                if isinstance(h.get("danger_reasoning"), str):
                    h["danger_reasoning"] = _strip_meta_preface(h.get("danger_reasoning", "")).strip()

            # Track extracted IDs to prevent duplicates in subsequent samples
            for h in reason_item.get("hazards", []):
                if isinstance(h, dict) and h.get("object_id") and h.get("object_id") not in extracted_ids:
                    extracted_ids.append(h["object_id"])

            # 2. Compute mean logprob for this reason
            tokens = list(choice.logprobs.content or [])
            token_logps = [float(getattr(tok, "logprob", -100.0)) for tok in tokens]
            mean_lp = float(np.mean(token_logps)) if token_logps else -10.0

            reasons_data.append(reason_item)
            mean_lps.append(mean_lp)

        if not reasons_data:
            return []

        # Normalize weights via softmax
        lp_arr = np.array(mean_lps, dtype=float)
        lp_shift = lp_arr - lp_arr.max()
        exp_lp = np.exp(lp_shift)
        p_reasons = (exp_lp / exp_lp.sum()).tolist()

        return [(json.dumps(d.get("hazards", []), ensure_ascii=False), float(w)) for d, w in zip(reasons_data, p_reasons)]

    # ------------------------------------------------------------------
    # Step 2 — Synthesized Scoring (Smarter Fusion)
    # ------------------------------------------------------------------

    def _score_synthesized(
        self,
        scene_block:  str,
        reasons:      list[tuple[str, float]],
        score_scale:  list[int],
        multimodal_items: list[dict] = None,
    ) -> tuple[list[dict[int, float]], list[str], str]:
        """
        Score each reason independently with a dedicated model call.
        Returns a probability distribution over the score scale for each reason.
        """
        n = len(reasons)
        score_dists: list[dict[int, float]] = []
        critiques: list[str] = []
        notes: list[str] = []

        for idx, reason in enumerate(reasons):
            prompt = _build_reason_scoring_prompt(scene_block, reason, idx)
            # Use the retry helper for LLM call and JSON extraction
            response, payload = self._call_llm_with_retry(
                prompt = prompt,
                temperature = 0.2,
                multimodal_items = multimodal_items,
                logprobs = True,
                top_logprobs = 10,
            )

            if response is None:
                # Handle total failure case
                dist = {s: 0.0 for s in score_scale}
                dist[3] = 1.0
                critique = "Total failure to get response."
                note = f"reason {idx}: total failure"
                score_dists.append(dist)
                critiques.append(critique)
                notes.append(note)
                continue

            raw_content = response.choices[0].message.content.strip()
            predicted_text = _strip_thinking_blocks(raw_content)
            try:
                _log_world_model_call(f"reason_scoring_{idx}", prompt, response)
            except Exception:
                logger.exception("Failed to log reason_scoring_%s response", idx)

            critique = "No critique provided (fallback)."
            note = ""
            dist = {s: 0.0 for s in score_scale}

            if payload:
                # JSON found and parsed by the helper
                json_probs = payload.get("score_probs")
                if isinstance(json_probs, dict):
                    extracted_probs = {}
                    for s_str, p_val in json_probs.items():
                        try:
                            s_int = int(s_str)
                            if s_int in score_scale:
                                extracted_probs[s_int] = float(p_val)
                        except (ValueError, TypeError):
                            continue

                    if len(extracted_probs) > 0:
                        total = sum(extracted_probs.values())
                        if total > 1.0:
                            # Weighted normalization: reduce all probabilities by a constant delta
                            # until the sum is 1.0. This ensures larger probabilities are reduced
                            # proportionally less and the peak is preserved better.
                            sorted_p = sorted(extracted_probs.values())
                            n = len(sorted_p)

                            # Find the correct number of elements that remain positive after subtracting delta
                            delta = 0.0
                            for k in range(1, n + 1):
                                # Sum of the k largest elements
                                sum_k_largest = sum(sorted_p[n-k:])
                                current_delta = (sum_k_largest - 1.0) / k
                                if k == n or sorted_p[n-k-1] <= current_delta:
                                    delta = current_delta
                                    break

                            for s, p in extracted_probs.items():
                                dist[s] = max(0.0, p - delta)
                            note = f"reason {idx}: distribution weighted-normalized from JSON (sum > 1)"
                        elif total > 0:
                            # Normalization: Ensure probabilities are strictly in [0, 1] and sum to 1.0.
                            # If total < 1.0, we scale them up. If total > 1.0, we scale them down.
                            scale = 1.0 / total
                            for s, p in extracted_probs.items():
                                dist[s] = float(p * scale)
                            note = f"reason {idx}: distribution linearly normalized from JSON (sum={total:.3f})"
                        else:
                            note = f"reason {idx}: JSON probabilities zero"

                critique = str(payload.get("critique") or critique)
            else:
                # JSON not found after 5 retries, fallback to Logprobs
                try:
                    content_logprobs = response.choices[0].logprobs.content
                    if content_logprobs:
                        first_token_lp = content_logprobs[0]
                        top_lps = first_token_lp.top_logprobs

                        found_lps = {}
                        for lp_item in top_lps:
                            t = lp_item.token.strip()
                            if t in map(str, score_scale):
                                found_lps[int(t)] = lp_item.logprob

                        if found_lps:
                            all_lps = [found_lps.get(s, -20.0) for s in score_scale]
                            shift_lps = np.array(all_lps) - np.max(all_lps)
                            exps = np.exp(shift_lps)
                            probs = exps / np.sum(exps)

                            for s, p in zip(score_scale, probs):
                                dist[s] = float(p)
                            note = f"reason {idx}: distribution from first token"
                        else:
                            score_token_idx = -1
                            for i, tok in enumerate(content_logprobs):
                                if tok.token.strip() in map(str, score_scale):
                                    score_token_idx = i
                                    break

                            if score_token_idx != -1:
                                top_lps = content_logprobs[score_token_idx].top_logprobs
                                found_lps = {int(lp.token.strip()): lp.logprob
                                             for lp in top_lps if lp.token.strip() in map(str, score_scale)}
                                all_lps = [found_lps.get(s, -20.0) for s in score_scale]
                                shift_lps = np.array(all_lps) - np.max(all_lps)
                                exps = np.exp(shift_lps)
                                probs = exps / np.sum(exps)
                                for s, p in zip(score_scale, probs):
                                    dist[s] = float(p)
                                note = f"reason {idx}: distribution from scanned token"
                            else:
                                dist[3] = 1.0
                                note = f"reason {idx}: neutral prior fallback"
                    else:
                        dist[3] = 1.0
                        note = f"reason {idx}: neutral prior fallback"
                except Exception as e:
                    logger.debug("Failed to extract score distribution: %s", e)
                    dist[3] = 1.0

            if not critique or critique == "No critique provided (fallback).":
                critique = predicted_text[:200].strip() or "Grounded using scene evidence and the mechanism described in the interpretation."

            score_dists.append(dist)
            critiques.append(critique)
            notes.append(note)

        combined_note = "; ".join([n for n in notes if n]) or "Per-reason scoring used: one model call per reason."
        return score_dists, critiques, combined_note

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _compute_final_risk_score(
        self,
        reasons:      list[tuple[str, float]],
        score_dists:   list[dict[int, float]],
        agents:        list[DetectedAgent],
        score_scale:   list[int],
        agent_hazards: dict[str, Hazard] = None,
    ) -> tuple[float, dict[int, float], list[dict]]:
        """
        Fuses per-reason score distributions into a general risk score using
        a severity-weighted softmax approach.
        """
        # 1. Basic Weights and Expectations
        raw_weights = []
        reason_expectations = []
        reason_object_ids = []

        for (text, p), dist in zip(reasons, score_dists):
            # Extract object_id
            oid = "unknown"
            try:
                parsed = json.loads(text)
                hazards = parsed if isinstance(parsed, list) else parsed.get("hazards", [])
                if hazards and isinstance(hazards[0], dict):
                    oid = hazards[0].get("object_id", "unknown")
            except Exception:
                pass

            # Environment fallback
            if oid == "unknown" or oid not in [a.agent_id for a in agents]:
                if any(k in text.lower() for k in ["road", "surface", "weather", "rain", "fog", "lighting"]):
                    oid = "environment"
            
            reason_object_ids.append(oid)
            
            # w_i = P(r_i) * distance_weight
            dist_w = self._get_distance_weight(oid, agents, agent_hazards=agent_hazards)
            raw_weights.append(p * dist_w)
            
            # E_i = Σ (score * prob)
            e_si = sum(s * prob for s, prob in dist.items())
            reason_expectations.append(e_si)

        # 2. Severity-Weighted Softmax
        T = _SOFTMAX_TEMP
        T = max(T, 1e-6)

        if not reason_expectations:
            return 1.0, {s: (1.0 if s == score_scale[0] else 0.0) for s in score_scale}, []

        # Numerically stable softmax: subtract max(E/T)
        e_over_t = np.array(reason_expectations) / T
        max_val = np.max(e_over_t)
        
        # w_i' = (w_i * exp(E_i/T)) / Σ (w_j * exp(E_j/T))
        exp_terms = np.exp(e_over_t - max_val)
        weighted_exp = np.array(raw_weights) * exp_terms
        sum_weighted_exp = np.sum(weighted_exp)

        if sum_weighted_exp > 0:
            softmax_weights = weighted_exp / sum_weighted_exp
        else:
            # Fallback to uniform if all are zero
            softmax_weights = np.full(len(raw_weights), 1.0 / len(raw_weights))

        # 3. Final Risk Score R = Σ (w_i' * E_i)
        general_risk_score = np.sum(softmax_weights * np.array(reason_expectations))

        # 4. Aggregate Distribution P(s) = Σ (w_i' * P(s|r_i))
        aggregate_score_probs = {s: 0.0 for s in score_scale}
        for sw, dist in zip(softmax_weights, score_dists):
            for s, prob in dist.items():
                aggregate_score_probs[s] += sw * prob

        # 5. Enrich reasons detail
        reasons_detail = []
        for i, ((text, p), dist) in enumerate(zip(reasons, score_dists)):
            hazards = []
            try:
                parsed = json.loads(text)
                hazards = parsed if isinstance(parsed, list) else parsed.get("hazards", [])
                if hazards and isinstance(hazards[0], dict):
                    hazards[0]["object_id"] = reason_object_ids[i]
            except Exception:
                hazards = [{
                    "object_id": reason_object_ids[i],
                    "hazard_type": "Collision Risk",
                    "location_relative": "unknown",
                    "observation": text,
                    "danger_reasoning": "N/A",
                    "actionable_risk": "Maintain defensive speed.",
                }]

            reasons_detail.append({
                "hazards": hazards,
                "softmax_weight": round(float(softmax_weights[i]), 4),
                "expected_score": round(reason_expectations[i], 3),
                "score_probs": {sk: round(prob, 4) for sk, prob in dist.items()},
            })

        return float(general_risk_score), aggregate_score_probs, reasons_detail

    def assess_epoch(
        self,
        segmented_items: list[list[dict]],
        env_description: str,
        frames:          any = None,
        depth_maps:      any = None,
        telemetry:       Optional[TelemetryData] = None,
        score_scale:     list[int] = [1, 2, 3, 4, 5],
        epoch_id:        str = "latest",
    ) -> dict:
        """
        Processes all frames in segmented_items together to capture temporal dynamics.
        """
        if not segmented_items:
            return {
                "expected_risk_score": 1.0,
                "final_score": 1.0,
                "score_probabilities": {s: (1.0 if s == 1 else 0.0) for s in score_scale},
                "reasons": [],
                "fallback_notes": ["Empty segmented items batch provided."],
                "context_summary": {"frame_count": 0},
            }

        depth_min, depth_max = compute_depth_bounds(segmented_items)
        epoch_agents = [
            build_agents_from_segments(frame_segs, depth_min=depth_min, depth_max=depth_max)
            for frame_segs in segmented_items
        ]
        all_agents_flat = [a for frame in epoch_agents for a in frame]

        # --- Risk Map Integration ---
        agent_risks = {}
        agent_hazards = {}
        if frames and depth_maps:
            # Use the last frame of the epoch as the reference for current risk
            last_idx = len(segmented_items) - 1

            # We handle both cubic (dict of lists) and non-cubic (list) formats
            if isinstance(frames, dict):
                for side in frames.keys():
                    img = frames[side][last_idx]
                    dmap = depth_maps[side][1][last_idx]
                    # Get agents for this frame that are present on this side
                    frame_agents = [a for a in epoch_agents[last_idx] if side in a.fused_sides]
                    if frame_agents:
                        risks, hazards = self._calculate_agent_risk_values(frame_agents, img, dmap, side=side)
                        agent_risks.update(risks)
                        agent_hazards.update(hazards)
            else:
                img = frames[last_idx]
                dmap = depth_maps["front"][1][last_idx]
                frame_agents = epoch_agents[last_idx]
                risks, hazards = self._calculate_agent_risk_values(frame_agents, img, dmap)
                agent_risks.update(risks)
                agent_hazards.update(hazards)

        # Build context for the reason sampler
        scene_context = {
            "is_epoch": True,
            "env_description": env_description,
            "epoch_agents": epoch_agents,
            "frames": frames,
            "telemetry": telemetry,
            "agent_risks": agent_risks,
            "epoch_id": epoch_id
        }

        # Initial multimodal content for the first sample
        first_block, multimodal_content = _build_epoch_scene_block(
            env_description, epoch_agents, frames, telemetry, agent_risks=agent_risks, epoch_id=epoch_id
        )
        scene_block = first_block

        if len(multimodal_content) > 10:
            multimodal_content = multimodal_content[:10]

        valid_ids = [a.agent_id for a in all_agents_flat]
        id_to_class = {a.agent_id: a.class_name for a in all_agents_flat}
        reasons = self._generate_reasons_with_logprobs(scene_context, multimodal_content, valid_ids=valid_ids, id_to_class=id_to_class)

        if not reasons:
            sides_present = sorted(set(a.side for a in all_agents_flat))
            closest_agent = min(all_agents_flat, key=lambda a: a.mode_depth) if all_agents_flat else None
            closest_info = (
                f"{closest_agent.class_name} @ depth {closest_agent.mode_depth:.3f} "
                f"(frame: {closest_agent.frame}, side: {closest_agent.side}, proximity: {closest_agent.distance_weight*100:.0f}%)"
                if closest_agent else "none"
            )
            return {
                "general_risk_score": float(score_scale[0]),
                "score_probabilities": {s: (1.0 if s == score_scale[0] else 0.0) for s in score_scale},
                "reasons": [],
                "fallback_notes": ["No reasons were generated by the model. Falling back to minimal risk."],
                "context_summary": {
                    "epoch_frames":         len(segmented_items),
                    "total_agents_detected": len(all_agents_flat),
                    "sides_observed":       sides_present,
                    "closest_agent":        closest_info,
                    "telemetry_available":  telemetry is not None,
                    "n_reasons":            self.n_reasons,
                },
            }

        score_dists, critiques, fallback_note = self._score_synthesized(
            scene_block, reasons, score_scale, multimodal_content
        )

        # New severity-weighted softmax fusion
        general_risk_score, aggregate_score_probs, reasons_detail = self._compute_final_risk_score(
            reasons, score_dists, all_agents_flat, score_scale, agent_hazards=agent_hazards
        )

        # Merge critiques back into reasons_detail
        for i, crit in enumerate(critiques):
            if i < len(reasons_detail):
                # The hazards array might be empty or have one item
                if reasons_detail[i]["hazards"]:
                    reasons_detail[i]["hazards"][0]["danger_reasoning"] = crit
                else:
                    reasons_detail[i]["critique"] = crit

        fallback_notes = [fallback_note] if fallback_note else []
        sides_present = sorted(set(a.side for a in all_agents_flat))
        closest_agent = min(all_agents_flat, key=lambda a: a.mode_depth) if all_agents_flat else None
        closest_info = (
            f"{closest_agent.class_name} @ depth {closest_agent.mode_depth:.3f} "
            f"(frame: {closest_agent.frame}, side: {closest_agent.side}, proximity: {closest_agent.distance_weight*100:.0f}%)"
            if closest_agent else "none"
        )

        return {
            "general_risk_score": round(general_risk_score, 3),
            "score_probabilities": {s: round(p, 4) for s, p in aggregate_score_probs.items()},
            "reasons":             reasons_detail,
            "fallback_notes":      fallback_notes,
            "context_summary": {
                "epoch_frames":         len(segmented_items),
                "total_agents_detected": len(all_agents_flat),
                "sides_observed":       sides_present,
                "closest_agent":        closest_info,
                "telemetry_available":  telemetry is not None,
                "n_reasons":            self.n_reasons,
            },
        }

    def evaluate_g_eval_risk(
        self,
        env_description: str,
        agents:          list[DetectedAgent],
        frames:          any = None,
        depth_map:       any = None,
        telemetry:       Optional[TelemetryData] = None,
        score_scale:     list[int] = [1, 2, 3, 4, 5],
    ) -> dict:
        """
        Full G-VEval hybrid pipeline for a single frame or set of pre-built DetectedAgents.
        """
        # --- Risk Map Integration ---
        agent_risks = {}
        agent_hazards = {}
        if frames and depth_map:
            # For a single frame, we can compute risk directly
            # Handle cubic (dict) vs non-cubic (single image)
            if isinstance(frames, dict):
                # This is rare for a single frame evaluation but handled for consistency
                for side in frames.keys():
                    img = frames[side][0] if isinstance(frames[side], list) else frames[side]
                    dmap = depth_map[side][0] if isinstance(depth_map[side], list) else depth_map[side]
                    side_agents = [a for a in agents if side in a.fused_sides]
                    if side_agents:
                        risks, hazards = self._calculate_agent_risk_values(side_agents, img, dmap, side=side)
                        agent_risks.update(risks)
                        agent_hazards.update(hazards)
            else:
                risks, hazards = self._calculate_agent_risk_values(agents, frames, depth_map)
                agent_risks.update(risks)
                agent_hazards.update(hazards)

        # Build context for the reason sampler
        scene_context = {
            "is_epoch": False,
            "env_description": env_description,
            "agents": agents,
            "frames": frames,
            "telemetry": telemetry,
            "agent_risks": agent_risks,
        }

        # Initial multimodal content for the first sample
        _, multimodal_content = _build_scene_block(
            env_description, agents, frames, telemetry, agent_risks=agent_risks
        )

        if len(multimodal_content) > 10:
            multimodal_content = multimodal_content[:10]

        valid_ids = [a.agent_id for a in agents]
        id_to_class = {a.agent_id: a.class_name for a in agents}
        reasons = self._generate_reasons_with_logprobs(scene_context, multimodal_content, valid_ids=valid_ids, id_to_class=id_to_class)

        if not reasons:
            sides_present = sorted(set(a.side for a in agents))
            closest_agent = min(agents, key=lambda a: a.mode_depth) if agents else None
            closest_info = (
                f"{closest_agent.class_name} @ depth {closest_agent.mode_depth:.3f} "
                f"(side: {closest_agent.side}, proximity: {closest_agent.distance_weight*100:.0f}%)"
                if closest_agent else "none"
            )
            return {
                "general_risk_score": float(score_scale[0]),
                "score_probabilities": {s: (1.0 if s == score_scale[0] else 0.0) for s in score_scale},
                "reasons": [],
                "fallback_notes": ["No reasons were generated by the model. Falling back to minimal risk."],
                "context_summary": {
                    "agents_count":        len(agents),
                    "sides_observed":      sides_present,
                    "closest_agent":       closest_info,
                    "telemetry_available": telemetry is not None,
                    "n_reasons":           self.n_reasons,
                },
            }

        score_dists, critiques, fallback_note = self._score_synthesized(
            scene_block, reasons, score_scale, multimodal_content
        )

        # New severity-weighted softmax fusion
        general_risk_score, aggregate_score_probs, reasons_detail = self._compute_final_risk_score(
            reasons, score_dists, agents, score_scale, agent_hazards=agent_hazards
        )

        # Merge critiques back into reasons_detail
        for i, crit in enumerate(critiques):
            if i < len(reasons_detail):
                if reasons_detail[i]["hazards"]:
                    reasons_detail[i]["hazards"][0]["danger_reasoning"] = crit
                else:
                    reasons_detail[i]["critique"] = crit

        fallback_notes = [fallback_note] if fallback_note else []

        sides_present = sorted(set(a.side for a in agents))
        closest_agent = min(agents, key=lambda a: a.mode_depth) if agents else None
        closest_info  = (
            f"{closest_agent.class_name} @ depth {closest_agent.mode_depth:.3f} "
            f"(side: {closest_agent.side}, proximity: {closest_agent.distance_weight*100:.0f}%)"
            if closest_agent else "none"
        )

        return {
            "general_risk_score": round(general_risk_score, 3),
            "score_probabilities": {s: round(p, 4) for s, p in aggregate_score_probs.items()},
            "reasons":             reasons_detail,
            "fallback_notes":      fallback_notes,
            "context_summary": {
                "agents_count":        len(agents),
                "sides_observed":      sides_present,
                "closest_agent":       closest_info,
                "telemetry_available": telemetry is not None,
                "n_reasons":           self.n_reasons,
            },
        }

    def assess_frame(
        self,
        frame_segments:  list[dict],
        env_description: str,
        frames:          any = None,
        telemetry:       Optional[TelemetryData] = None,
        depth_min:       float = 0.0,
        depth_max:       float = 1.0,
        score_scale:     list[int] = [1, 2, 3, 4, 5],
    ) -> dict:
        return self.assess_epoch(
            segmented_items = [frame_segments],
            env_description = env_description,
            frames = frames,
            telemetry       = telemetry,
            score_scale     = score_scale,
        )
