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

  Step 2 — Conditional scoring via logprobs  (N calls, temperature = 0)
      For each reason rᵢ, the model evaluates P(s | scene, rᵢ) by reading the
      log-probabilities over the candidate score tokens {1, 2, 3}.  This gives a
      sub-score per reason:

          sub_score(rᵢ) = Σ  s · P(s | scene, rᵢ)
                         s ∈ {1,2,3}

  Step 3 — Weighted fusion (G-VEval outer summation)
      The final continuous risk score is the expectation over reasons:

          E[score] = Σᵢ  P(rᵢ) · sub_score(rᵢ)

      where P(rᵢ) is obtained by softmax-normalising the mean per-token log-
      probability of each reason across the N samples.

      This produces a genuinely continuous value (e.g. 2.66 instead of 2) that
      enables fine-grained tie-breaking and threshold alerting.

References
----------
  Liu et al. (2023) "G-Eval: NLG Evaluation using GPT-4 with Better Human
  Alignment."  ACL 2023.  https://aclanthology.org/2023.acl-long.244

  G-VEval (2024) "Fine-Grained Risk Assessment."
  arXiv 2412.13647  https://arxiv.org/pdf/2412.13647
"""

import re
import math
import cv2
import base64
import numpy as np
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Optional

from src.tasks.config.utils import CONFIG

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
# Each criterion is (name, description).  They are injected into both the
# reason-generation prompt and the scoring form so the model evaluates the
# same axes in both steps.
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
        "and riders are more vulnerable and less predictable than cars or "
        "trucks.  Their presence at high proximity should raise the score.",
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
    telemetry:       Optional[TelemetryData] = None,
) -> str:
    """
    Assembles a temporal scene context block across a sequence/batch of frames
    (an "epoch" sliding window). Shows agent trajectory dynamics, depth trends,
    and multi-camera coverage across the epoch.
    """
    total_frames = len(epoch_agents)
    all_agents_flat = [a for frame in epoch_agents for a in frame]
    sides_present = sorted(set(a.side for a in all_agents_flat))

    # Group agents across the epoch by class and camera side to trace trajectory
    tracks: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
    for frame_idx, frame in enumerate(epoch_agents):
        for a in frame:
            key = (a.class_name, a.side)
            if key not in tracks:
                tracks[key] = []
            tracks[key].append((frame_idx, a.mode_depth, a.distance_weight * 100))

    if tracks:
        track_lines = []
        for (cls_name, side), occurrences in tracks.items():
            frames_seen = [occ[0] for occ in occurrences]
            proximities = [occ[2] for occ in occurrences]
            min_prox = min(proximities)
            max_prox = max(proximities)
            first_depth = occurrences[0][1]
            last_depth = occurrences[-1][1]

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
                f"  • Track [{cls_name} on {side}] ({frame_span_str})\n"
                f"    - Proximity Range: {min_prox:.0f}% to {max_prox:.0f}% (Peak: {max_prox:.0f}%)\n"
                f"    - Temporal Trend: {trend}"
            )
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

    return (
        f"[Epoch Temporal Context]\n"
        f"Sliding Window / Epoch Batch: {total_frames} frames  |  Camera sides observed: {', '.join(sides_present) or 'none'}\n\n"
        f"[Environment]\n{env_description}\n\n"
        f"[Agent Trajectories & Temporal Trends across Epoch]\n"
        f"{epoch_agents_block}\n\n"
        f"[Physical Telemetry]\n{telemetry_block}"
    )


def _build_scene_block(
    env_description: str,
    agents:          list[DetectedAgent],
    telemetry:       Optional[TelemetryData],
) -> str:
    """
    Assembles the shared scene-context block for single-frame evaluation.
    """
    sides_present = sorted(set(a.side for a in agents))
    if agents:
        agent_lines = []
        for a in agents:
            fused_tag = " [fused]" if a.was_fused else ""
            agent_lines.append(
                f"  • [{a.agent_id}] {a.class_name}{fused_tag}"
                f"  |  side: {a.side}"
                f"  |  depth: {a.mode_depth:.3f}"
                f"  |  proximity: {a.distance_weight * 100:.0f}%"
                f"  |  seg-conf: {a.seg_score * 100:.0f}%"
            )
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

    return (
        f"[Environment]\n{env_description}\n\n"
        f"[Detected Agents]  (camera sides observed: {', '.join(sides_present) or 'none'})\n"
        f"{agents_block}\n\n"
        f"[Physical Telemetry]\n{telemetry_block}"
    )


def _build_reason_prompt(scene_block: str) -> str:
    """
    Step 1 prompt — asks the model to produce ONE focused, polar risk-interpretation
    by analyzing physical safety margins (TTC, proximity) and counterfactuals.
    """
    criteria_text = "\n".join(
        f"  {i+1}. {name}: {desc}"
        for i, (name, desc) in enumerate(EVALUATION_CRITERIA)
    )

    return (
        "You are an expert road-safety auditor evaluating crash risk for a motorcycle rider "
        "equipped with a 360° perception system.\n\n"
        "=== TASK OVERVIEW ===\n"
        "Analyze the temporal scene data below. Your goal is to identify the SINGLE MOST "
        "critical risk trajectory or confirm the complete absence of physical hazards.\n\n"
        "=== EVALUATION CRITERIA ===\n"
        f"{criteria_text}\n\n"
        "=== SCENE DATA (EPOCH BATCH) ===\n"
        f"{scene_block}\n\n"
        "=== MANDATORY ANALYSIS STEPS ===\n"
        "1. KINEMATIC CHECK: Identify the minimum Time-To-Collision (TTC), sudden deceleration/acceleration "
        "   (>3 m/s²), or aggressive lateral shifts (cut-ins/encroachments) across all agent tracks.\n"
        "2. COUNTERFACTUAL BOUNDARY TEST:\n"
        "   - Is this scenario SAFE (Minimal Risk)? Explain why no agent trajectory intersects or threatens the rider.\n"
        "   - OR is this scenario CRITICAL (High/Severe Risk)? Explain what immediate evasive action or physical "
        "     hazard pushes this beyond routine driving.\n"
        "3. POLARITY FOCUS: Do NOT summarize the scene as 'mildly cautious' or 'moderate.' Take a definitive stand "
        "   on whether the scene leans clearly safe or clearly hazardous based on physical spatial margins.\n\n"
        "=== OUTPUT INSTRUCTIONS ===\n"
        "Write 1 concise risk interpretation (3-4 sentences). Reference specific track IDs, exact depth/distance "
        "values, and velocity/TTC figures. Do NOT assign a score number.\n\n"
        "Risk Interpretation:"
    )


def _build_scoring_prompt(scene_block: str, reason: str) -> str:
    """
    Step 2 prompt — Enforces strict score mapping using physical thresholds (TTC, hard maneuvers)
    and explicit rubric boundaries to prevent central tendency bias.
    """
    return (
        "You are a strict motorcycle risk scoring model. Evaluate the scene data and risk interpretation "
        "and assign an exact risk score from 1 to 5.\n\n"
        "=== SCENE DATA ===\n"
        f"{scene_block}\n\n"
        "=== RISK INTERPRETATION ===\n"
        f"{reason.strip()}\n\n"
        "=== RIGID ANCHOR RUBRIC ===\n"
        "Assign the score based strictly on these physical hazard anchors:\n"
        "• 1 (Minimal Risk): Free flow traffic, clear spatial buffers (>5s TTC), standard cruising. No threat.\n"
        "• 2 (Low Risk): Minor density or routine speed adjustments. Normal awareness required; no path conflicts.\n"
        "• 3 (Moderate Risk): Noticeable interaction (TTC 3-5s), gradual deceleration ahead, or mild lateral encroachment. "
        "  *ONLY use 3 if physical metrics strictly fall in this middle zone.*\n"
        "• 4 (High Risk): Abrupt hazard, hard braking (>3 m/s²), aggressive cut-in, or severe buffer loss (TTC 2-3s). "
        "  Rider must actively adjust trajectory/speed.\n"
        "• 5 (Severe / Critical Risk): Immediate collision threat or trajectory overlap (TTC < 2s). Emergency evasive "
        "  action (hard braking/swerving) is required to avoid a crash.\n\n"
        "=== SCORING RULES ===\n"
        "1. Do NOT default to 3 out of caution. If there is low interaction, score 1 or 2. If there is a close cut-in or hard brake, score 4 or 5.\n"
        "2. If TTC < 2 seconds or a direct trajectory conflict exists, you MUST score 4 or 5.\n\n"
        "Output ONLY a single digit integer (1, 2, 3, 4, or 5) corresponding to the risk score.\n"
        "Score:"
    )


# ---------------------------------------------------------------------------
# 4. Risk Assessment Engine
# ---------------------------------------------------------------------------

class RiskAssessmentEngine:
    """
    G-VEval risk assessment engine for the 360° motorcycle safety pipeline.

    Three-stage pipeline per frame
    --------------------------------
    1. ``_generate_reasons_with_logprobs()`` — N independent reason-sampling calls
    2. ``_score_with_logprobs()``            — logprob-weighted scoring per reason
    3. Weighted outer summation              — G-VEval double expectation

    Final score formula
    -------------------
        E[score] = Σᵢ P(rᵢ) · Σₛ s · P(s | scene, rᵢ)

    where P(rᵢ) is the softmax-normalised mean per-token log-probability of
    each reason sampled in Stage 1.

    Primary entry points
    --------------------
    ``evaluate_g_eval_risk()``  — accepts pre-built DetectedAgent list
    ``assess_frame()``          — accepts raw segment dicts from process.py
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

    # ------------------------------------------------------------------
    # Step 1 — Reason sampling with logprob weights
    # ------------------------------------------------------------------

    def _generate_reasons_with_logprobs(
        self,
        scene_block: str,
    ) -> list[tuple[str, float]]:
        """
        Calls the model N times independently (temperature > 0) to obtain N
        diverse risk interpretations of the scene.  For each call we read the
        per-token log-probabilities and compute the mean log-probability as a
        proxy for how "natural / confident" that interpretation is.

        The N mean log-probabilities are then converted to a proper probability
        distribution via softmax, yielding P(rᵢ).

        Args
        ----
        scene_block : assembled scene context (from _build_scene_block).

        Returns
        -------
        list of (reason_text, p_reason) tuples, sorted by p_reason descending.
        p_reason values sum to 1.0.
        """
        prompt = _build_reason_prompt(scene_block)

        reason_texts: list[str]   = []
        mean_logprobs: list[float] = []

        for _ in range(self.n_reasons):
            response = self.client.chat.completions.create(
                model       = self.model_name,
                messages    = [{"role": "user", "content": prompt}],
                # Higher temperature encourages diverse risk angles
                temperature = 0.7,
                max_tokens  = 128,
                logprobs    = True,
            )

            choice = response.choices[0]
            reason_texts.append(choice.message.content.strip())

            # Compute mean per-token log-probability for this reason
            token_logprobs = [
                tok.logprob
                for tok in (choice.logprobs.content or [])
                if tok.logprob is not None
            ]
            mean_lp = (
                float(np.mean(token_logprobs)) if token_logprobs else -10.0
            )
            mean_logprobs.append(mean_lp)

        # Softmax over mean log-probabilities → P(rᵢ)
        lp_arr   = np.array(mean_logprobs, dtype=float)
        lp_shift = lp_arr - lp_arr.max()          # numerical stability
        exp_lp   = np.exp(lp_shift)
        p_reasons = (exp_lp / exp_lp.sum()).tolist()

        return list(zip(reason_texts, p_reasons))

    # ------------------------------------------------------------------
    # Step 2 — Logprob-weighted scoring conditioned on one reason
    # ------------------------------------------------------------------

    def _score_with_logprobs(
        self,
        scene_block:  str,
        reason:       str,
        score_scale:  list[int],
    ) -> tuple[dict[int, float], str]:
        """
        Calls the model with temperature=0 and logprobs=True, conditioned on
        a single risk interpretation (reason). Reads the log-probabilities of
        the generated token and maps them onto the candidate score digits
        to compute the conditional sub-score P(s | scene, reason).
        """
        prompt = _build_scoring_prompt(scene_block, reason)

        response = self.client.chat.completions.create(
            model        = self.model_name,
            messages     = [{"role": "user", "content": prompt}],
            temperature  = 0.0,  # greedy — deterministic scoring conditioned on reason
            max_tokens   = 4,
            logprobs     = True,
            top_logprobs = 10,   # top-10 maximises chance of seeing all 5 score tokens
        )

        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs

        # Convert log-probabilities to natural probabilities for score tokens
        raw_probs: dict[int, float] = {}
        for item in top_logprobs:
            token = item.token.strip()
            clean_token = token.lstrip()
            if clean_token and clean_token[0].isdigit():
                digit = int(clean_token[0])
                if digit in score_scale:
                    raw_probs[digit] = max(raw_probs.get(digit, 0.0), math.exp(item.logprob))

        score_probs   = {s: 0.0 for s in score_scale}
        fallback_note = ""
        total         = sum(raw_probs.values())

        if total > 0:
            # Normalise so probabilities over valid tokens sum to 1
            for s, p in raw_probs.items():
                score_probs[s] = p / total
        else:
            # Fall back to regex parsing on greedy generated output
            predicted = response.choices[0].message.content.strip()
            match = re.search(r'\b([1-5])\b', predicted) or re.search(r'([1-5])', predicted)
            if match:
                fallback = int(match.group(1))
                if fallback not in score_scale:
                    fallback = score_scale[len(score_scale) // 2]
                fallback_note = (
                    f"logprobs did not contain raw score token; "
                    f"extracted score {fallback} from greedy output '{predicted}'"
                )
            else:
                fallback = score_scale[len(score_scale) // 2]  # median fallback (3)
                fallback_note = (
                    f"logprobs and greedy output '{predicted}' lacked valid score digit; "
                    f"fell back to median score {fallback}"
                )
            score_probs[fallback] = 1.0

        return score_probs, fallback_note

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_epoch(
        self,
        segmented_items: list[list[dict]],
        env_description: str,
        telemetry:       Optional[TelemetryData] = None,
        score_scale:     list[int] = [1, 2, 3, 4, 5],
    ) -> dict:
        """
        G-VEval risk assessment across an epoch (sliding window / batch of frames).
        Processes all frames in segmented_items together to capture temporal dynamics,
        agent approach trends, and cumulative risk across the epoch window.
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

        scene_block = _build_epoch_scene_block(env_description, epoch_agents, telemetry)
        reasons = self._generate_reasons_with_logprobs(scene_block)

        expected_risk_score = 0.0
        fused_score_probs = {s: 0.0 for s in score_scale}
        reasons_detail: list[dict] = []
        fallback_notes: list[str] = []

        for reason_text, p_reason in reasons:
            score_probs, fallback_note = self._score_with_logprobs(
                scene_block, reason_text, score_scale
            )
            sub_score = sum(s * p for s, p in score_probs.items())
            expected_risk_score += p_reason * sub_score

            for s in score_scale:
                fused_score_probs[s] += p_reason * score_probs[s]

            reasons_detail.append({
                "text":        reason_text,
                "weight":      round(p_reason, 4),
                "sub_score":   round(sub_score, 4),
                "score_probs": {s: round(p, 4) for s, p in score_probs.items()},
            })
            if fallback_note:
                fallback_notes.append(f"[reason '{reason_text[:40]}…'] {fallback_note}")

        all_agents_flat = [a for frame in epoch_agents for a in frame]
        sides_present = sorted(set(a.side for a in all_agents_flat))
        closest_agent = min(all_agents_flat, key=lambda a: a.mode_depth) if all_agents_flat else None
        closest_info = (
            f"{closest_agent.class_name} @ depth {closest_agent.mode_depth:.3f} "
            f"(frame: {closest_agent.frame}, side: {closest_agent.side}, proximity: {closest_agent.distance_weight*100:.0f}%)"
            if closest_agent else "none"
        )

        return {
            "expected_risk_score": round(expected_risk_score, 3),
            "final_score":         round(expected_risk_score, 3),
            "score_probabilities": {s: round(p, 4) for s, p in fused_score_probs.items()},
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
        telemetry:       Optional[TelemetryData] = None,
        score_scale:     list[int] = [1, 2, 3, 4, 5],
    ) -> dict:
        """
        Full G-VEval pipeline for a single frame or set of pre-built DetectedAgents.
        """
        scene_block = _build_scene_block(env_description, agents, telemetry)
        reasons = self._generate_reasons_with_logprobs(scene_block)

        expected_risk_score   = 0.0
        fused_score_probs     = {s: 0.0 for s in score_scale}
        reasons_detail: list[dict] = []
        fallback_notes: list[str]  = []

        for reason_text, p_reason in reasons:
            score_probs, fallback_note = self._score_with_logprobs(
                scene_block, reason_text, score_scale
            )
            sub_score = sum(s * p for s, p in score_probs.items())
            expected_risk_score += p_reason * sub_score

            for s in score_scale:
                fused_score_probs[s] += p_reason * score_probs[s]

            reasons_detail.append({
                "text":        reason_text,
                "weight":      round(p_reason, 4),
                "sub_score":   round(sub_score, 4),
                "score_probs": {s: round(p, 4) for s, p in score_probs.items()},
            })
            if fallback_note:
                fallback_notes.append(f"[reason '{reason_text[:40]}…'] {fallback_note}")

        sides_present = sorted(set(a.side for a in agents))
        closest_agent = min(agents, key=lambda a: a.mode_depth) if agents else None
        closest_info  = (
            f"{closest_agent.class_name} @ depth {closest_agent.mode_depth:.3f} "
            f"(side: {closest_agent.side}, proximity: {closest_agent.distance_weight*100:.0f}%)"
            if closest_agent else "none"
        )

        return {
            "final_score":         round(expected_risk_score, 3),
            "score_probabilities": {s: round(p, 4) for s, p in fused_score_probs.items()},
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
        telemetry:       Optional[TelemetryData] = None,
        depth_min:       float = 0.0,
        depth_max:       float = 1.0,
        score_scale:     list[int] = [1, 2, 3, 4, 5],
    ) -> dict:
        """
        Convenience entry point: takes raw segment dicts for a single frame.
        """
        return self.assess_epoch(
            segmented_items = [frame_segments],
            env_description = env_description,
            telemetry       = telemetry,
            score_scale     = score_scale,
        )
