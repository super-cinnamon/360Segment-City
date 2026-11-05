"""
VLM Hazard Identification Module
================================

Stage 1 of the paper pipeline: VLM-based hazard identification with
motorcycle state context (speed, lean angle).
Queries the active VLM (gemma-4-26b-a4b) to get hazard list + c_vlm scores.
"""

import json
import logging
import re
from typing import List, Dict, Optional, Any
import numpy as np
import cv2

from src.tasks.environment import query_world_model
from src.tasks.config.utils import CONFIG, resolve_vlm_profile

logger = logging.getLogger(__name__)


# Default prompt template (also saved as file for easy editing)
# Use {{ and }} to escape curly braces in the JSON example
DEFAULT_HAZARD_IDENTIFICATION_PROMPT = """You are an expert motorcycle safety system. Analyze the front-facing camera image and the motorcycle's current state.

MOTORCYCLE STATE:
- Speed: {speed_kmh} km/h
- Lean Angle: {lean_angle_deg} degrees

Identify ALL potential hazards that pose a risk to the motorcycle. For each hazard, provide:
1. A descriptive label (e.g., "pothole", "wet road surface", "debris on road", "car cutting in", "pedestrian crossing")
2. A contextual risk score c_vlm in [0, 1] representing how dangerous this hazard is GIVEN the current speed and lean angle
3. A brief description of the hazard mechanism
4. Hazard type: "dynamic_agent", "environment_hazard", or "road_condition"

Output ONLY valid JSON:
{{
  "hazards": [
    {{
      "label": "pothole",
      "c_vlm": 0.85,
      "description": "Deep pothole in lane center, unavoidable at current lean angle",
      "hazard_type": "road_condition"
    }},
    ...
  ]
}}

If no hazards are detected, return: {{"hazards": []}}"""


class VLMHazardIdentifier:
    """
    VLM-based hazard identification with motorcycle state context.

    Implements Stage 1 of the paper pipeline:
    - Input: RGB frame + motorcycle state (speed, lean angle) + prompt
    - Output: List of hazards with c_vlm scores
    """

    def __init__(self, model_name: Optional[str] = None, prompt_template: Optional[str] = None):
        """
        Initialize VLM hazard identifier.

        Args:
            model_name: VLM model name (uses active VLM from config if None)
            prompt_template: Custom prompt template (uses default if None)
        """
        # Use active VLM from config
        if model_name is None:
            profile = resolve_vlm_profile()
            self.model_name = profile["repo_id"]
        else:
            self.model_name = model_name

        self.prompt_template = prompt_template or DEFAULT_HAZARD_IDENTIFICATION_PROMPT
        self.last_raw_response = ""
        logger.info(f"VLMHazardIdentifier initialized with model: {self.model_name}")

    def _build_prompt(self, speed_kmh: float, lean_angle_deg: float) -> str:
        """Build the prompt with motorcycle state for logging."""
        return self.prompt_template.format(
            speed_kmh=speed_kmh,
            lean_angle_deg=lean_angle_deg
        )

    def identify_hazards(
        self,
        image: np.ndarray,
        speed_kmh: float,
        lean_angle_deg: float
    ) -> List[Dict]:
        """
        Query VLM with frame + motorcycle state to get hazards + c_vlm scores.

        Args:
            image: Front-facing RGB frame (H, W, 3)
            speed_kmh: Current motorcycle speed in km/h
            lean_angle_deg: Current lean angle in degrees

        Returns:
            List of dicts with keys:
                - label: hazard label string
                - c_vlm: contextual risk score [0, 1]
                - description: hazard mechanism description
                - hazard_type: "dynamic_agent" | "environment_hazard" | "road_condition"
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to VLM hazard identifier")
            return []

        # Build prompt with motorcycle state
        prompt = self.prompt_template.format(
            speed_kmh=speed_kmh,
            lean_angle_deg=lean_angle_deg
        )

        # Query VLM
        try:
            # Use existing query_world_model infrastructure
            # It supports single image input
            response = query_world_model(
                prompt=prompt,
                images=[image],
                model=self.model_name
            )

            # Store raw response for logging
            self.last_raw_response = response

            logger.debug(f"VLM hazard identification response: {response[:500]}...")

            # Parse JSON response
            hazards = self._parse_vlm_response(response)

            logger.info(f"VLM identified {len(hazards)} hazards: {[h['label'] for h in hazards]}")
            return hazards

        except Exception as e:
            logger.error(f"VLM hazard identification failed: {e}")
            self.last_raw_response = f"ERROR: {e}"
            return []

    def _parse_vlm_response(self, response: str) -> List[Dict]:
        """Parse VLM response to extract hazards list."""
        hazards = []

        # Try to extract JSON from response
        json_str = self._extract_json(response)
        if not json_str:
            logger.warning("No JSON found in VLM response")
            return []

        try:
            data = json.loads(json_str)
            raw_hazards = data.get("hazards", [])

            for h in raw_hazards:
                if not isinstance(h, dict):
                    continue

                # Validate required fields
                label = h.get("label", "").strip()
                if not label:
                    continue

                c_vlm = float(h.get("c_vlm", 0.0))
                c_vlm = max(0.0, min(1.0, c_vlm))  # Clamp to [0, 1]

                description = h.get("description", "").strip()
                hazard_type = h.get("hazard_type", "environment_hazard")

                # Validate hazard_type
                valid_types = ["dynamic_agent", "environment_hazard", "road_condition"]
                if hazard_type not in valid_types:
                    hazard_type = "environment_hazard"

                hazards.append({
                    "label": label,
                    "c_vlm": c_vlm,
                    "description": description,
                    "hazard_type": hazard_type
                })

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse VLM JSON response: {e}")
        except Exception as e:
            logger.warning(f"Error processing VLM hazards: {e}")

        return hazards

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text response."""
        # Try to find JSON object (handles markdown code blocks)
        # First try: look for ```json ... ``` pattern
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Second try: look for first { ... } balanced
        # Find first {
        start = text.find('{')
        if start == -1:
            return None

        # Find matching }
        brace_count = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i+1]

        return None


def create_vlm_hazard_identifier(
    model_name: Optional[str] = None,
    prompt_template: Optional[str] = None
) -> VLMHazardIdentifier:
    """Factory function to create VLM hazard identifier."""
    return VLMHazardIdentifier(model_name=model_name, prompt_template=prompt_template)