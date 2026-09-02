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
import logging
import cv2
import base64
import numpy as np
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Optional

from src.tasks.config.utils import CONFIG

# Module logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level config aliases
# ---------------------------------------------------------------------------
_RA_CFG           = CONFIG["risk_assessment"]
_DEFAULT_API_BASE = _RA_CFG["api_base"]
_DEFAULT_MODEL    = _RA_CFG["model_name"]
_DEFAULT_N        = int(_RA_CFG.get("n_reasons", 5))

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
    a_ego     : Ego-motorcycle longitudinal acceleration (m/s²) — from IMU
    roll      : Roll angle (degrees)               — from IMU
    yaw_rate  : Yaw rate (deg/s)                   — from IMU
    """
    ttc:      float
    pet:      float
    d_lat:    float
    v_rel:    float
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
    mask:            Optional[np.ndarray] = field(default=None, repr=False)
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

    for idx, seg in enumerate(frame_segments):
        raw_depth        = float(seg.get("mode_depth", 0.5))
        clamped          = max(depth_min, min(depth_max, raw_depth))
        normalized_depth = (clamped - depth_min) / depth_range
        distance_weight  = round(1.0 - normalized_depth, 4)

        agent_id = (
            f"{seg.get('side', 'unk')}_"
            f"frame{seg.get('frame', 0)}_"
            f"{seg['class_name'].replace(' ', '_')}_"
            f"{idx}"
        )

        agents.append(DetectedAgent(
            agent_id        = agent_id,
            class_name      = seg["class_name"],
            side            = seg.get("side", "unknown"),
            frame           = seg.get("frame", 0),
            seg_score       = float(seg.get("score") or 0.0),
            mode_depth      = raw_depth,
            distance_weight = distance_weight,
            mask            = seg.get("mask"),
            was_fused       = seg.get("was_fused", False),
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
    all_depths = [
        seg["mode_depth"]
        for frame in segmented_items
        for seg in frame
        if seg.get("mode_depth") is not None
    ]
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
) -> tuple[str, dict[str, str]]:
    """
    Assembles a temporal scene context block across a sequence/batch of frames
    (an "epoch" sliding window). Shows agent trajectory dynamics, depth trends,
    and multi-camera coverage across the epoch.
    """
    total_frames = len(epoch_agents)
    all_agents_flat = [a for frame in epoch_agents for a in frame]
    sides_present = sorted(set(a.side for a in all_agents_flat))

    # Group agents across the epoch by class and camera side to trace trajectory
    tracks: dict[tuple[str, str], list[DetectedAgent]] = {}
    for frame_idx, frame in enumerate(epoch_agents):
        for a in frame:
            key = (a.class_name, a.side)
            if key not in tracks:
                tracks[key] = []
            tracks[key].append(a)

    image_map = {}
    if tracks:
        track_lines = []
        for (cls_name, side), occurrences in tracks.items():
            frames_seen = [a.frame for a in occurrences]
            proximities = [a.distance_weight * 100 for a in occurrences]
            min_prox = min(proximities)
            max_prox = max(proximities)
            first_depth = occurrences[0].mode_depth
            last_depth = occurrences[-1].mode_depth

            if len(occurrences) > 1:
                depth_delta = last_depth - first_depth
                if depth_delta < -0.05:
                    trend = "APPROACHING / CLOSING IN (HIGHER DANGER)"
                elif depth_delta > 0.05:
                    trend = "RECEDING / MOVING AWAY"
                else:
                    trend = "STATIONARY / CONSTANT DISTANCE"
            else:
                trend = "TRANSIENT DETECTION"

            frame_span_str = (
                f"frames {min(frames_seen)}..{max(frames_seen)}"
                if len(frames_seen) > 1 else f"frame {frames_seen[0]}"
            )
            track_lines.append(
                f"  • Track [ID: {occurrences[0].agent_id}] [{cls_name} on {side}] ({frame_span_str})\n"
                f"    - Proximity Range: {min_prox:.0f}% to {max_prox:.0f}% (Peak: {max_prox:.0f}%)\n"
                f"    - Temporal Trend: {trend}"
            )

            # Extract image for the peak proximity frame
            peak_agent = max(occurrences, key=lambda a: a.distance_weight)
            try:
                f_idx = peak_agent.frame
                s = peak_agent.side
                mask = peak_agent.mask
                if frames:
                    frame_img = frames[s][f_idx] if isinstance(frames, dict) else frames[f_idx]
                    image_map[peak_agent.agent_id] = _crop_agent_image(frame_img, mask)
            except Exception:
                pass

            # Only the first occurrence ID is used for the track key usually,
            # but here we use the actual agent_id for images.
            # Actually, since we grouped by (class, side), we can just use the
            # first occurrence's ID to refer to the whole track in the prompt.
            # Let's use the first occurrence ID for the image_map too if we want 1 image per track.
            # But let's use the peak agent's ID just to be safe.

        epoch_agents_block = "\n".join(track_lines)
    else:
        epoch_agents_block = "  (no dynamic agents detected across epoch)"

    if telemetry is not None:
        telemetry_block = (
            f"  TTC={telemetry.ttc:.2f}s  |  PET={telemetry.pet:.2f}s  |  "
            f"v_rel={telemetry.v_rel:.1f} m/s  |  d_lat={telemetry.d_lat:.1f} m  |  "
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
    return text, image_map


def _build_scene_block(
    env_description: str,
    agents:          list[DetectedAgent],
    frames:          any,
    telemetry:       Optional[TelemetryData],
) -> tuple[str, dict[str, str]]:
    """
    Assembles the shared scene-context block for single-frame evaluation.
    """
    sides_present = sorted(set(a.side for a in agents))
    image_map = {}
    if agents:
        agent_lines = []
        for a in agents:
            fused_tag = " [fused]" if a.was_fused else ""
            agent_lines.append(
                f"  • [ID: {a.agent_id}] {a.class_name}{fused_tag}"
                f"  |  side: {a.side}"
                f"  |  depth: {a.mode_depth:.3f}"
                f"  |  proximity: {a.distance_weight * 100:.0f}%"
                f"  |  seg-conf: {a.seg_score * 100:.0f}%"
            )
            # Extract image for this agent
            try:
                f_idx = a.frame
                s = a.side
                mask = a.mask
                if frames:
                    frame_img = frames[s][f_idx] if isinstance(frames, dict) else frames[f_idx]
                    image_map[a.agent_id] = _crop_agent_image(frame_img, mask)
            except Exception:
                pass

        agents_block = "\n".join(agent_lines)
    else:
        agents_block = "  (no dynamic agents detected)"

    if telemetry is not None:
        telemetry_block = (
            f"  TTC={telemetry.ttc:.2f}s  |  PET={telemetry.pet:.2f}s  |  "
            f"v_rel={telemetry.v_rel:.1f} m/s  |  d_lat={telemetry.d_lat:.1f} m  |  "
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
    return text, image_map


def _build_reason_prompt(scene_block: str) -> str:
    """
    Step 1 prompt — asks the model to produce a concise, actionable dynamic hazard
    in a defensive-driving JSON format.
    """
    criteria_text = "\n".join(
        f"  {i+1}. {name}: {desc}"
        for i, (name, desc) in enumerate(EVALUATION_CRITERIA)
    )

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
        "GROUNDING RULES FOR THE HAZARD:\n"
        "- observation: one concrete, physically grounded sentence describing what is visible or moving. Use terms like\n"
        "  'left-side sedan', 'wet metal seam', 'narrowing gap', 'approaching cyclist', or 'braking lead vehicle'.\n"
        "  Mention specific high-risk details: if it is a child, if the agent is facing the road, or if turn signals are active.\n"
        "- danger_reasoning: explain the mechanism in 2-3 detailed clauses: how it threatens the rider, why it matters now "
        "  (e.g., 'child may dart into road', 'facing road suggests imminent entry'), and what specific action is needed. "
        "  Be explicit about vulnerability and unpredictable intent. Do not mention being an AI or discussing the prompt.\n"
        "- actionable_risk: give imperative defensive-driving guidance such as 'Reduce speed and hold the left edge' or\n"
        "  'Prepare a controlled swerve to the right'.\n"
        "- You must produce EXACTLY ONE most-significant hazard. If no grounded threat exists, return an empty array [].\n\n"
        "EVALUATION FRAMEWORK (Analyze hazards across these 7 categories):\n",
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
        "3 = MODERATE RISK — Elevated caution required; one or more agents are close or behaving unpredictably; a speed or position adjustment may be needed.\n"
        "4 = HIGH RISK     — Significant hazard; braking or evasive manoeuvre likely required soon; collision window is opening.\n"
        "5 = SEVERE RISK   — Imminent hazard; emergency manoeuvre required right now or collision is unavoidable.\n\n"
        "JUDGING RULES:\n"
        "1. Be critical: if the hazard description implies an immediate threat, do not default to 3. Use 4 or 5.\n"
        "2. Be grounded: if the scene data contradicts the interpretation, score it lower.\n"
        "3. Differentiate: avoid assigning a 'neutral' 3 if the evidence strongly points to a specific rubric level.\n"
        "4. Amplify Vulnerability: Scenarios involving children, pedestrians/cyclists in close proximity, or "
        "   unpredictable intent (e.g., facing the road without moving) should strongly push the score toward 4 or 5.\n"
        "5. Amplify Awareness & Signal Risk: Scenarios where a hazard is closing in while the rider appears unaware (e.g., from a blind spot), or where critical traffic signals (red lights/stop signs) or lane markings are being ignored, should strongly push the score toward 4 or 5.\n\n",
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

# ! problematic approach
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


def _humanize_hazards(hazard_payload: str | dict) -> str:
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
        lines.append(f"Hazard {i+1}: [{htype}] {obs} (obj:{oid}, loc:{loc})")
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
    # try:
    #     msg_p = f"\n{'='*80}\n[PROMPT: {label}]\n{prompt}\n{'='*80}"
    #     logger.info(msg_p)
    #     print(msg_p)
    # except Exception:
    #     pass

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


# ---------------------------------------------------------------------------
# 4. Risk Assessment Engine
# ---------------------------------------------------------------------------

# Constant for environmental hazard distance weight
ENVIRONMENT_DISTANCE_WEIGHT = 0.5

class RiskAssessmentEngine:
    """
    Hybrid risk assessment engine for the 360° motorcycle safety pipeline.

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

    def _get_distance_weight(self, object_id: str, agents: list[DetectedAgent]) -> float:
        """Retrieves distance weight for an object; defaults to ENVIRONMENT_DISTANCE_WEIGHT."""
        if not object_id or object_id == "unknown":
            return ENVIRONMENT_DISTANCE_WEIGHT

        if object_id == "environment":
            return ENVIRONMENT_DISTANCE_WEIGHT

        for a in agents:
            if a.agent_id == object_id:
                return a.distance_weight

        return ENVIRONMENT_DISTANCE_WEIGHT

    def _call_llm_with_retry(
        self,
        prompt: str,
        temperature: float,
        images: Optional[list[str]] = None,
        max_tokens: int = 8192,
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
                if images:
                    for img_b64 in images:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
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
        scene_block: str,
        image_map:   dict[str, str] = None,
    ) -> list[tuple[str, float]]:
        """
        Calls the model N times independently (temperature > 0) to obtain N
        diverse risk interpretations of the scene.

        Calculates the weight P(r_i) for each reason using the softmax-normalized
        mean per-token log-probabilities.
        """
        prompt = _build_reason_prompt(scene_block)
        reasons_data = []
        mean_lps = []

        # Extract image list from map
        images = list(image_map.values()) if image_map else None

        for i in range(self.n_reasons):
            # Use the retry helper for LLM call and JSON extraction
            response, reason_item = self._call_llm_with_retry(
                prompt = prompt,
                temperature = 0.9,
                images = images,
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
                if isinstance(h.get("observation"), str):
                    h["observation"] = _strip_meta_preface(h.get("observation", "")).strip()
                if isinstance(h.get("danger_reasoning"), str):
                    h["danger_reasoning"] = _strip_meta_preface(h.get("danger_reasoning", "")).strip()

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
        image_map:    dict[str, str] = None,
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
                images = list(image_map.values()) if image_map else None,
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
                            for s, p in extracted_probs.items():
                                dist[s] = p
                            note = f"reason {idx}: distribution from JSON"
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

    def assess_epoch(
        self,
        segmented_items: list[list[dict]],
        env_description: str,
        frames:          any = None,
        telemetry:       Optional[TelemetryData] = None,
        score_scale:     list[int] = [1, 2, 3, 4, 5],
    ) -> dict:
        """
        Processes all frames in segmented_items together to capture temporal dynamics.

        Final Score Calculation (Distance-Aware G-VEval Fusion):
        1. Reasons are sampled with confidence P(r_i).
        2. Each reason is scored as a distribution D_i.
        3. Weights are computed as W_raw,i = P(r_i) * distance_weight(object_id).
        4. Weights are normalized: W_final,i = W_raw,i / Sum(W_raw).
        5. Final Score = Sum(W_final,i * ExpectedScore(D_i)).
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

        scene_block, image_map = _build_epoch_scene_block(env_description, epoch_agents, frames, telemetry)
        reasons = self._generate_reasons_with_logprobs(scene_block, image_map)

        # HYBRID FUSION: Get score distributions and critiques for each reason
        score_dists, critiques, fallback_note = self._score_synthesized(
            scene_block, reasons, score_scale, image_map
        )

        # --- Weight Calculation & Normalization ---
        raw_weights = []
        reason_object_ids = []

        for (text, p), _ in zip(reasons, score_dists):
            # Extract object_id from reason JSON
            oid = "unknown"
            try:
                parsed = json.loads(text)
                hazards = parsed if isinstance(parsed, list) else parsed.get("hazards", [])
                if hazards and isinstance(hazards[0], dict):
                    oid = hazards[0].get("object_id", "unknown")
            except Exception:
                pass

            # Ensure environmental hazards are explicitly marked
            if oid == "unknown" or oid not in [a.agent_id for a in all_agents_flat]:
                # If the reason text mentions environment keywords, mark as environment
                if any(k in text.lower() for k in ["road", "surface", "weather", "rain", "fog", "lighting"]):
                    oid = "environment"

            reason_object_ids.append(oid)
            dist_w = self._get_distance_weight(oid, all_agents_flat)
            raw_weights.append(p * dist_w)

        sum_raw = sum(raw_weights) if raw_weights else 1.0
        final_weights = [rw / sum_raw if sum_raw > 0 else 1.0/max(1, len(raw_weights)) for rw in raw_weights]

        # --- Final Score & Aggregate Distribution ---
        general_risk_score = 0.0
        aggregate_score_probs = {s: 0.0 for s in score_scale}
        reason_expectations = []

        for w, dist in zip(final_weights, score_dists):
            # Expected score for this reason: Sum(k * p_k)
            e_si = sum(s * p for s, p in dist.items())
            reason_expectations.append(e_si)
            general_risk_score += w * e_si

            # Aggregate distribution: P(score=k) = Sum(W_final,i * P(score=k | r_i))
            for s, p in dist.items():
                aggregate_score_probs[s] += w * p

        # Enrich reasons detail
        reasons_detail = []
        for i, ((text, p), dist, crit) in enumerate(zip(reasons, score_dists, critiques)):
            hazards = []
            try:
                parsed = json.loads(text)
                hazards = parsed if isinstance(parsed, list) else parsed.get("hazards", [])
                # Update object_id to "environment" if we determined it was environmental
                if hazards and isinstance(hazards[0], dict):
                    hazards[0]["object_id"] = reason_object_ids[i]
            except Exception:
                hazards = [{
                    "object_id": reason_object_ids[i],
                    "hazard_type": "Collision Risk",
                    "location_relative": "unknown",
                    "observation": text,
                    "danger_reasoning": crit,
                    "actionable_risk": "Maintain defensive speed and reassess the scene.",
                }]

            reasons_detail.append({
                "hazards": hazards,
                "weight": round(final_weights[i], 4),
                "expected_score": round(reason_expectations[i], 3),
                "score_probs": {sk: round(prob, 4) for sk, prob in dist.items()},
            })

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
        telemetry:       Optional[TelemetryData] = None,
        score_scale:     list[int] = [1, 2, 3, 4, 5],
    ) -> dict:
        """
        Full G-VEval hybrid pipeline for a single frame or set of pre-built DetectedAgents.

        Final Score Calculation (Distance-Aware G-VEval Fusion):
        Final Score = Σ [W_final,i * ExpectedScore(D_i)]
        where W_final,i is the normalized product of reason confidence and object distance.
        """
        scene_block, image_map = _build_scene_block(env_description, agents, frames, telemetry)
        reasons = self._generate_reasons_with_logprobs(scene_block, image_map)

        # HYBRID FUSION: Get score distributions and critiques for each reason
        score_dists, critiques, fallback_note = self._score_synthesized(
            scene_block, reasons, score_scale, image_map
        )

        # --- Weight Calculation & Normalization ---
        raw_weights = []
        reason_object_ids = []

        for (text, p), _ in zip(reasons, score_dists):
            oid = "unknown"
            try:
                parsed = json.loads(text)
                hazards = parsed if isinstance(parsed, list) else parsed.get("hazards", [])
                if hazards and isinstance(hazards[0], dict):
                    oid = hazards[0].get("object_id", "unknown")
            except Exception:
                pass

            if oid == "unknown" or oid not in [a.agent_id for a in agents]:
                if any(k in text.lower() for k in ["road", "surface", "weather", "rain", "fog", "lighting"]):
                    oid = "environment"

            reason_object_ids.append(oid)
            dist_w = self._get_distance_weight(oid, agents)
            raw_weights.append(p * dist_w)

        sum_raw = sum(raw_weights) if raw_weights else 1.0
        final_weights = [rw / sum_raw if sum_raw > 0 else 1.0/max(1, len(raw_weights)) for rw in raw_weights]

        # --- Final Score & Aggregate Distribution ---
        general_risk_score = 0.0
        aggregate_score_probs = {s: 0.0 for s in score_scale}
        reason_expectations = []

        for w, dist in zip(final_weights, score_dists):
            e_si = sum(s * p for s, p in dist.items())
            reason_expectations.append(e_si)
            general_risk_score += w * e_si
            for s, p in dist.items():
                aggregate_score_probs[s] += w * p

        reasons_detail = []
        for i, ((text, p), dist, crit) in enumerate(zip(reasons, score_dists, critiques)):
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
                    "danger_reasoning": crit,
                    "actionable_risk": "Maintain defensive speed and reassess the scene.",
                }]

            reasons_detail.append({
                "hazards": hazards,
                "weight": round(final_weights[i], 4),
                "expected_score": round(reason_expectations[i], 3),
                "score_probs": {sk: round(prob, 4) for sk, prob in dist.items()},
            })

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
