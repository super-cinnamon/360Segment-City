import base64
import math
import cv2
import numpy as np
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Optional

from src.tasks.config.utils import CONFIG

# Convenience aliases so callers don't have to index CONFIG themselves
_RA_CFG = CONFIG["risk_assessment"]
_DEFAULT_API_BASE = _RA_CFG["api_base"]
_DEFAULT_MODEL    = _RA_CFG["model_name"]

# ---------------------------------------------------------------------------
# 1. Configuration & Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TelemetryData:
    """
    Optional physics-sensor telemetry (e.g. from IMUs).
    All fields represent measured or derived quantities at the time of assessment.
    Not required for the pipeline to function — pass None where TelemetryData
    is accepted to skip the telemetry section of the G-Eval prompt.
    """
    # Derived safety metrics (can be computed from IMU + GPS fusion)
    ttc: float          # Time To Collision (seconds)
    pet: float          # Post Encroachment Time (seconds)
    d_lat: float        # Lateral Distance to nearest agent (meters)
    v_rel: float        # Relative Speed to nearest agent (m/s)
    # IMU-derived motion state
    a_ego: float        # Ego-motorcycle longitudinal acceleration (m/s²)
    roll: float         # Roll angle (degrees)
    yaw_rate: float     # Yaw rate (deg/s)


@dataclass
class DetectedAgent:
    """
    Represents a single segmented object from one camera side/frame,
    built directly from the dicts produced by SegmentationPipeline.process_vision().

    segment dict schema (from process.py):
        {
            "frame":      int,
            "side":       str,          # "front" | "left" | "right" | "back"
            "class_name": str,
            "class_id":   int,
            "score":      float | None, # segmentation confidence
            "was_fused":  bool,
            "mask":       np.ndarray,   # binary boolean mask
            "mode_depth": float,        # modal depth value; LOWER = CLOSER = HIGHER RISK
        }
    """
    agent_id: str           # Unique identifier, e.g. "front_frame2_car_0"
    class_name: str         # Human-readable class label from id2label
    side: str               # Camera side this agent was detected on
    frame: int              # Frame index within the processed batch
    seg_score: float        # Segmentation confidence score (0.0–1.0)
    mode_depth: float       # Raw modal depth value (lower = closer to camera)
    distance_weight: float  # Proximity weight derived from depth: 1 - norm(mode_depth).
                            # Range [0.0, 1.0] — higher means closer / more dangerous.
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    was_fused: bool = False


# ---------------------------------------------------------------------------
# 2. Factory: Build DetectedAgent list from process.py output
# ---------------------------------------------------------------------------

def build_agents_from_segments(
    frame_segments: list[dict],
    depth_min: float = 0.0,
    depth_max: float = 1.0,
) -> list[DetectedAgent]:
    """
    Converts a single frame's segment list (as returned by
    SegmentationPipeline.process_vision()) into a list of DetectedAgent objects.

    Depth normalization:
        mode_depth values across the batch are normalized into [0, 1] so that
        distance_weight = 1 - normalized_depth, giving the closest objects the
        highest proximity weight (most dangerous).

    Args:
        frame_segments: List of segment dicts for a single frame (all sides).
        depth_min:  Minimum depth value observed across the *full batch* of frames,
                    used for consistent normalization. Defaults to 0.0 (per-frame).
        depth_max:  Maximum depth value observed across the *full batch* of frames.
                    Defaults to 1.0 (per-frame).

    Returns:
        List of DetectedAgent objects, one per segment item.
    """
    agents = []
    depth_range = depth_max - depth_min if depth_max > depth_min else 1.0

    for idx, seg in enumerate(frame_segments):
        raw_depth = float(seg.get("mode_depth", 0.5))

        # Clamp depth to [depth_min, depth_max] then normalize to [0, 1]
        clamped = max(depth_min, min(depth_max, raw_depth))
        normalized_depth = (clamped - depth_min) / depth_range

        # Invert: lower depth (closer) → higher distance_weight → higher danger
        distance_weight = 1.0 - normalized_depth

        agent_id = f"{seg.get('side', 'unk')}_frame{seg.get('frame', 0)}_{seg['class_name'].replace(' ', '_')}_{idx}"

        agents.append(DetectedAgent(
            agent_id=agent_id,
            class_name=seg["class_name"],
            side=seg.get("side", "unknown"),
            frame=seg.get("frame", 0),
            seg_score=float(seg.get("score") or 0.0),
            mode_depth=raw_depth,
            distance_weight=round(distance_weight, 4),
            mask=seg.get("mask"),
            was_fused=seg.get("was_fused", False),
        ))

    return agents


def compute_depth_bounds(segmented_items: list[list[dict]]) -> tuple[float, float]:
    """
    Scans all frames in segmented_items to find the global min/max mode_depth
    values, enabling consistent normalization across the full batch.

    Args:
        segmented_items: Full output of SegmentationPipeline.process_vision().

    Returns:
        (depth_min, depth_max) tuple.
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
# 3. Risk Assessment Engine
# ---------------------------------------------------------------------------

class RiskAssessmentEngine:
    """
    G-Eval based risk assessment engine for the 360° motorcycle safety pipeline.

    Accepts:
      - An environment description string (from environment.py VLM queries)
      - A list of DetectedAgent objects (built from process.py segmented_items)
      - An optional TelemetryData object (IMU / physics sensors — pass None if unavailable)

    Returns a continuous expected risk score via logprob-weighted expectation
    over discrete candidate score tokens {1, 2, 3}.
    """

    def __init__(
        self,
        api_base: str = _DEFAULT_API_BASE,
        model_name: str = _DEFAULT_MODEL,
    ):
        self.client = OpenAI(base_url=api_base, api_key="vllm-local")
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_image(self, cv2_img: np.ndarray) -> str:
        """Encodes an OpenCV BGR image buffer directly to a Base64 JPEG string."""
        _, buffer = cv2.imencode('.jpg', cv2_img)
        return base64.b64encode(buffer).decode('utf-8')

    def _format_agents_text(self, agents: list[DetectedAgent]) -> str:
        """Formats detected agents into a concise, human-readable block for the prompt."""
        if not agents:
            return "  (no dynamic agents detected in this frame)"

        lines = []
        for a in agents:
            fused_tag = " [fused]" if a.was_fused else ""
            lines.append(
                f"  - [{a.agent_id}] {a.class_name}{fused_tag} | "
                f"Side: {a.side} | "
                f"Depth: {a.mode_depth:.3f} | "
                f"Proximity Weight: {a.distance_weight * 100:.0f}% | "
                f"Seg. Confidence: {a.seg_score * 100:.0f}%"
            )
        return "\n".join(lines)

    def _format_telemetry_text(self, telemetry: Optional[TelemetryData]) -> str:
        """
        Formats telemetry into a compact string for the G-Eval prompt.
        Returns an explicit placeholder when telemetry is unavailable.
        """
        if telemetry is None:
            return "  (no telemetry available — physics sensor integration pending)"
        return (
            f"  TTC: {telemetry.ttc:.2f}s | "
            f"PET: {telemetry.pet:.2f}s | "
            f"Rel. Velocity: {telemetry.v_rel:.1f} m/s | "
            f"Lat. Distance: {telemetry.d_lat:.1f} m | "
            f"Roll: {telemetry.roll:.1f}° | "
            f"Yaw Rate: {telemetry.yaw_rate:.1f} °/s"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe_environment(self, image: np.ndarray, prompt: str) -> str:
        """
        Queries the VLM API endpoint for a structured environment description.
        This is a thin wrapper kept for compatibility — prefer using the dedicated
        query_world_model() / query_ollama_vlm() functions in environment.py
        which handle both Windows (HF) and Linux (vLLM) backends.

        Args:
            image:  OpenCV BGR image (typically the front-facing frame).
            prompt: The VLM prompt string (e.g. ENV_PROMPT from config).

        Returns:
            Raw text response from the VLM.
        """
        base64_image = self._encode_image(image)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=256,
        )
        return response.choices[0].message.content

    def evaluate_g_eval_risk(
        self,
        env_description: str,
        agents: list[DetectedAgent],
        telemetry: Optional[TelemetryData] = None,
        score_scale: list[int] = [1, 2, 3],
    ) -> dict:
        """
        G-Eval step: computes a continuous expected risk score by sampling
        log-probabilities over discrete candidate score tokens {1, 2, 3}.

        Context is assembled from:
          - env_description : structured JSON output from the VLM environment model
          - agents          : DetectedAgent list built from process.py segmented_items
          - telemetry       : optional TelemetryData (pass None if no IMU available)

        Args:
            env_description: String output from the VLM environment description step.
            agents:          List of DetectedAgent objects for the current frame.
            telemetry:       Optional TelemetryData. When None, the telemetry section
                             in the prompt explicitly notes the absence of sensor data.
            score_scale:     List of candidate integer score tokens. Defaults to [1, 2, 3].

        Returns:
            dict with keys:
                "expected_risk_score"  : float — continuous score in [1, 3]
                "score_probabilities"  : dict[int, float] — P(score=s) for each s
                "context_summary"      : dict — metadata about the evaluated frame
        """
        agents_text = self._format_agents_text(agents)
        telemetry_text = self._format_telemetry_text(telemetry)

        # Derive a lightweight per-frame summary for the context block
        sides_present = sorted(set(a.side for a in agents))
        closest_agent = min(agents, key=lambda a: a.mode_depth) if agents else None
        closest_info = (
            f"{closest_agent.class_name} @ depth {closest_agent.mode_depth:.3f} ({closest_agent.side})"
            if closest_agent else "none"
        )

        prompt = (
            f"=== 360° MOTORCYCLE RISK ASSESSMENT ===\n\n"
            f"[Environment Context]\n{env_description}\n\n"
            f"[Detected Dynamic Agents] (sides observed: {', '.join(sides_present) if sides_present else 'none'})\n"
            f"{agents_text}\n\n"
            f"[Physical Telemetry]\n{telemetry_text}\n\n"
            f"Evaluation Task:\n"
            f"Assess the overall immediate safety risk to the motorcycle rider based on the "
            f"environment description, the spatial proximity and type of all detected agents "
            f"(proximity weight 100% = closest, 0% = furthest), and any available telemetry.\n\n"
            f"Risk Score Definitions:\n"
            f"  1 = Low Risk    — Normal traffic conditions; safe distances; no imminent hazard.\n"
            f"  2 = Moderate Risk — Caution warranted; close agents, braking, or lane changes needed.\n"
            f"  3 = Severe Risk  — Imminent hazard; collision trajectory; emergency maneuver required.\n\n"
            f"Output ONLY the single integer digit (1, 2, or 3) representing the overall risk score.\n"
            f"Score:"
        )

        # Query vLLM-compatible API server requesting top logprobs for the score token
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
        )

        # Extract raw log-probabilities for the first generated token
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs

        # Map score digits to their natural probabilities
        score_probs = {s: 0.0 for s in score_scale}
        raw_logits: dict[int, float] = {}

        for item in top_logprobs:
            token_str = item.token.strip()
            if token_str.isdigit() and int(token_str) in score_scale:
                raw_logits[int(token_str)] = math.exp(item.logprob)

        # Softmax normalization across the valid score tokens found in top_logprobs
        total_prob_mass = sum(raw_logits.values())
        if total_prob_mass > 0:
            for score_val, p_raw in raw_logits.items():
                score_probs[score_val] = p_raw / total_prob_mass
        else:
            # Fallback: the model's greedy token was not in top_logprobs — use it directly
            predicted_text = response.choices[0].message.content.strip()
            fallback_score = int(predicted_text) if predicted_text.isdigit() and int(predicted_text) in score_scale else 2
            score_probs[fallback_score] = 1.0

        # G-Eval Expected Risk Score: E[Score] = Σ( s · P(s) )
        expected_risk_score = sum(s * prob for s, prob in score_probs.items())

        return {
            "expected_risk_score": round(expected_risk_score, 3),
            "score_probabilities": {s: round(p, 4) for s, p in score_probs.items()},
            "context_summary": {
                "agents_count": len(agents),
                "sides_observed": sides_present,
                "closest_agent": closest_info,
                "telemetry_available": telemetry is not None,
            },
        }

    def assess_frame(
        self,
        frame_segments: list[dict],
        env_description: str,
        telemetry: Optional[TelemetryData] = None,
        depth_min: float = 0.0,
        depth_max: float = 1.0,
        score_scale: list[int] = [1, 2, 3],
    ) -> dict:
        """
        Convenience method: takes a single frame's segment list directly from
        SegmentationPipeline.process_vision() and runs the full G-Eval pipeline.

        This is the primary entry point for integration with process.py.

        Args:
            frame_segments:  One element from segmented_items (list of segment dicts).
            env_description: VLM environment description string (from environment.py).
            telemetry:       Optional TelemetryData object. Pass None if no IMU available.
            depth_min:       Global depth minimum for normalization (from compute_depth_bounds).
            depth_max:       Global depth maximum for normalization (from compute_depth_bounds).
            score_scale:     Candidate score tokens. Defaults to [1, 2, 3].

        Returns:
            G-Eval risk assessment result dict (see evaluate_g_eval_risk).
        """
        agents = build_agents_from_segments(
            frame_segments,
            depth_min=depth_min,
            depth_max=depth_max,
        )
        return self.evaluate_g_eval_risk(
            env_description=env_description,
            agents=agents,
            telemetry=telemetry,
            score_scale=score_scale,
        )
