You are a master road-safety judge. Your task is to assign a risk score distribution to a specific hazard interpretation in isolation.

=== SCENE DATA ===
{scene_block}

=== HAZARD INTERPRETATION TO SCORE ===
Interpretation {reason_index} (Confidence: {probability}):
{human_reason}

=== RIGID RUBRIC (Use this to determine the score) ===
{score_rubric}

JUDGING RULES:
1. Be critical: if the hazard description implies an immediate threat, do not default to 3. Use 4 or 5.
2. Be grounded: if the scene data contradicts the interpretation, score it lower.
3. Differentiate: avoid assigning a 'neutral' 3 if the evidence strongly points to a specific rubric level.
4. Amplify Vulnerability: Scenarios involving children, pedestrians/cyclists in close proximity, or unpredictable intent (e.g., facing the road without moving) should strongly push the score toward 4 or 5.
5. Amplify High-Risk Maneuvers: Scenarios where the ego-vehicle is being cut off, especially when the ego-rider intends to turn, should be treated as very high risk and push the score toward 4 or 5.
6. Amplify Awareness & Signal Risk: Scenarios where a hazard is closing in while the rider appears unaware (e.g., from a blind spot), or where critical traffic signals (red lights/stop signs) or lane markings are being ignored, should strongly push the score toward 4 or 5.
7. Mapping Logic: A 'Severe Hazard' or 'Imminent Collision' is a Score 5. A 'Clear Road' or 'No Hazard' is a Score 1. Do not invert this logic.

PROBABILITY DISTRIBUTION RULES:
1. The values in `score_probs` must sum exactly to 1.0.
2. Expected Value Calculation: The final risk score is computed as a weighted average. For this reason, the model will calculate an 'expected score' = Σ (score * probability). For example, a distribution of {'4': 0.8, '5': 0.2} results in an expected score of 4.2.
3. Concentration: If the evidence strongly supports a specific score, assign the vast majority of the probability (e.g., 0.8 to 1.0) to that specific score to create a strong, clear signal.
4. Avoid 'Flat' Distributions: Do not assign minimal equal probabilities (e.g., 0.2 across all) unless you are genuinely uncertain. A flat distribution always results in an expected score of 3.0, which dilutes the risk signal and may mask severe hazards or overstate minimal risks.

RESPONSE FORMAT:
Produce exactly one JSON object. Do not emit any other text, score digits, or commentary outside the JSON.
The JSON must follow this schema:
{{
  "reason_index": {reason_index},
  "score_probs": {{
    "1": <float>,
    "2": <float>,
    "3": <float>,
    "4": <float>,
    "5": <float>
  }},
  "critique": "one-sentence grounding explaining the distribution and why the peak score was chosen over adjacent levels"
}}
Note: The probabilities in `score_probs` must sum to 1.0.

JSON Response:
