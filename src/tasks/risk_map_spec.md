# Specification: Risk Map Construction Module

**Scope**: Implementation of the spatial risk map construction (Algorithm 1, Lines 1–7). Motion Planning (Lines 8–13) is OUT OF SCOPE.

## 1. World Model Interface and Adapter [ASSUMPTION]

### 1.1 World Model Contract
*   **Model**: The system uses a VLM-based World Model (e.g., via `query_world_model` in `src/tasks/environment.py`).
*   **Inputs**:
    *   RGB Image(s).
    *   Bike State $s = [v, \phi]$.
*   **Output**: The model natively outputs a text response. An **Adapter** is required to parse this text into a structured JSON list: `[{"hazard": "pothole", "score": 8.5}, ...]`.
*   **Latency**: The model is queried **per frame**. The hazard set $\mathcal{H}_t$ is re-evaluated every frame.
*   **Score Scaling**: Raw scores are expected on a $0\text{--}10$ scale. These **must** be rescaled to $c_{\text{vlm}} \in [0,1]$ via $c_{\text{vlm}} = \text{score}_{\text{raw}} / 10.0$. `[ASSUMPTION]`

### 1.2 Hazard Filtering [ASSUMPTION]
To prevent noise from non-hazardous environmental objects, a hazard is only processed if it meets **one** of these criteria:
1.  **Allowlist**: The label is present in `dynamic_labels` (e.g., `pothole`, `puddle`, `gravel`, `oil`, `debris`, `cone`, `sign`, `wet_surface`).
2.  **Score Threshold**: The VLM assigned score $\ge \text{score\_threshold}$.
3.  **Synonym Mapping**: The label matches a synonym in the `hazard_synonyms` map (e.g., "oil slick" $\rightarrow$ "oil").

## 2. Risk Map Construction [PAPER]

For every filtered hazard $i \in \mathcal{H}_t$:

### 2.1 Grounding and Masking
*   **Grounded-SAM**: Generate a binary mask $M_i$ and a detection confidence $c_{\text{conf}}$.
*   **Validation**: A hazard is ignored if:
    *   The mask $M_i$ is empty.
    *   The mask area $|M_i| < \text{min\_mask\_area}$.
    *   The confidence $c_{\text{conf}} < \text{min\_conf\_threshold}$.

### 2.2 Multi-Factor Cost Calculation
The per-hazard scalar cost $C_i$ is calculated as:
$$C_i = \alpha_{\text{vlm}} c_{\text{vlm}} + \alpha_{\text{area}} c_{\text{area}} + \alpha_{\text{conf}} c_{\text{conf}} + \alpha_{\text{depth}} c_{\text{depth}}$$

*   **Semantic Risk ($c_{\text{vlm}}$)**: Normalized VLM score $\in [0,1]$.
*   **Area Risk ($c_{\text{area}}$)**: $\sigma(\kappa_a (\frac{|M_i|}{H \times W} - \eta_a))$.
*   **Confidence Risk ($c_{\text{conf}}$)**: Grounded-SAM detection confidence.
*   **Depth Risk ($c_{\text{depth}}$)**: $\min(1, \frac{d_i}{d_{\text{ref}}})$.
    *   **Computation of $d_i$ [ASSUMPTION]**:
        1.  Project depth pixels into 3D space using camera intrinsics $K$.
        2.  Fit a road plane to the depth pixels identified as "road" in the segmentation map using RANSAC.
        3.  $d_i$ is the **maximum signed deviation** (distance) of pixels within $M_i$ *below* the fitted road plane.
    *   **Exclusion [ASSUMPTION]**: The depth term ($\alpha_{\text{depth}} c_{\text{depth}}$) is **excluded** (and weights renormalized) if:
        *   The hazard is not "depth-sensitive" (e.g., oil, wet surface, signs).
        *   No valid depth map is available for the frame.

### 2.3 Spatial Aggregation
The final risk map $\mathcal{C}_{\text{risk}}$ is constructed by taking the pixel-wise maximum across all hazards:
$$\mathcal{C}_{\text{risk}}(m,n) = \max_{i \in \mathcal{H}_t} (C_i \text{ if } (m,n) \in M_i \text{ else } 0)$$

## 3. Weighting and Normalization [ASSUMPTION]

### 3.1 Default Configuration
*   **Weights**: $\alpha_{\text{vlm}}=0.4, \alpha_{\text{area}}=0.2, \alpha_{\text{conf}}=0.2, \alpha_{\text{depth}}=0.2$.
*   **Hyperparameters**: $\kappa_a = 75.0, \eta_a = 0.05, d_{\text{ref}} = 0.1\text{m}$.
*   **Global Max**: $C_{\text{max}} = 1.0$.

### 3.2 Dynamic Weight Renormalization [ASSUMPTION]
To maintain $C_i \in [0, 1]$, if any term is excluded (e.g., $c_{\text{depth}}$ for non-depth hazards or $\alpha_{\text{vlm}}=0$ for ablation), the remaining active weights $\alpha'_j$ are rescaled to sum to $1.0$:
$$\alpha'_j = \frac{\alpha_j}{\sum_{k \in \text{active}} \alpha_k}$$

## 4. Temporal Smoothing [ASSUMPTION]
To prevent risk "flickering" across frames, optional Exponential Moving Average (EMA) smoothing is applied to the per-hazard costs $C_i$ over time:
$$\hat{C}_{i,t} = \beta \hat{C}_{i,t-1} + (1-\beta) C_{i,t}$$
where $\beta \in [0,1]$ is the smoothing factor.

## 5. Failure Modes [ASSUMPTION]
*   **Zero Hazards**: $\mathcal{C}_{\text{risk}}$ is a zero matrix.
*   **Background**: Pixels not belonging to any hazard mask have a risk value of $0$.
*   **Normalization**: Data used for planning uses the fixed $C_{\text{max}}=1$. Normalization to $[0,1]$ is performed **only for display/visualization**.
