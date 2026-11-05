# SYSTEM PROMPT / SPECIFICATION: VLM-Based Motorcycle Hazard Risk Map Module

## 1. Overview
Implement the **Risk Map Construction** module of the VLM-Based Advanced Rider Assistance System (ARAS) for motorcycles based on the paper *VLM-Based Advanced Rider Assistance System for Motorcycle Safety*.

**Scope Note**: Implement ONLY the perception and Risk Map generation module (Section IV-A, IV-B, and Algorithm 1, Lines 1–6). Do NOT implement the sampling-based dynamic motion planner (Algorithm 1, Lines 7–12).

---

## 2. Mathematical Formulation & Equations

The goal is to compute a dense spatial risk map $\mathcal{C}_{risk} \in [0, \mathcal{C}_{max}]^{H \times W}$ for a given front-facing camera frame $I_{RGB} \in \mathbb{R}^{H \times W \times 3}$ and current motorcycle state $s_t = [v_t, \phi_t]$ (velocity $v_t$ and lean angle $\phi_t$).

### 2.1 Per-Hazard Multi-Factor Risk Cost
For each identified hazard $h_i \in \mathcal{H}_t$, assign a per-pixel risk cost $C_i(m, n)$ for all pixels $(m, n)$ inside its binary segmentation mask $\mathcal{M}_i \in \{0, 1\}^{H \times W}$:

$$C_i(m, n) = \alpha_{vlm} c_i^{vlm} + \alpha_{area} c_i^{area} + \alpha_{conf} c_i^{conf} + \alpha_{depth} c_i^{depth}$$

Where:
1. **Contextual VLM Risk ($c_i^{vlm} \in [0, 1]$)**: Derived from Vision-Language Model semantic reasoning over the image $I_{RGB}$, motorcycle state $s_t$, and prompt $T_{prompt}$:
   $$c_i^{vlm} = \text{VLM}(I_{RGB}, s_t, T_{prompt})$$

2. **Area-Based Risk ($c_i^{area} \in [0, 1]$)**: Sigmoid-normalized ratio of the hazard mask relative to total frame resolution:
   $$c_i^{area} = \sigma\left(\kappa_a \left( \frac{\vert{}\mathcal{M}_i\vert{}}{H \times W} - \eta_a \right)\right) = \frac{1}{1 + e^{-\kappa_a \left( \frac{\vert{}\mathcal{M}_i\vert{}}{H \times W} - \eta_a \right)}}$$

3. **Confidence Score ($c_i^{conf} \in [0, 1]$)**: Detection confidence $p_i$ produced by the open-vocabulary grounding/segmentation model (e.g., Grounded-SAM).

4. **Depth-Based Risk ($c_i^{depth} \in [0, 1]$)**: Normalized physical depth deviation $d_i$ relative to a reference depth threshold $d_{ref}$:
   $$c_i^{depth} = \min\left(1, \frac{d_i}{d_{ref}}\right)$$

### 2.2 Pixel-Wise Map Aggregation
When hazard masks overlap, select the maximum hazard severity across all $J$ identified hazards for every pixel location $(m, n)$:

$$\mathcal{C}_{risk}(m, n) = \max_{i=1}^{J} \mathcal{C}_i(m, n)$$

---

## 3. Algorithm Specification

```text
Algorithm: Risk Map Construction Module
--------------------------------------------------------------------------------
Input : RGB Frame I_{RGB, t}, Motorcycle State s_t = [v_t, \phi_t],
        Prompt T_{prompt}, Depth Map D_t (optional for d_i calculation)
Output: Dense Risk Map \mathcal{C}_{risk} \in \mathbb{R}^{H \times W}

1: Query VLM(I_{RGB, t}, s_t, T_{prompt}) -> Obtain hazard set \mathcal{H}_t = {h_i}
   and contextual risk scores {c_i^{vlm}}
2: For each hazard h_i \in \mathcal{H}_t do:
3:     Obtain binary mask \mathcal{M}_i and detection confidence c_i^{conf} using Grounded-SAM
4:     Compute area score c_i^{area} using Equation (3)
5:     Compute depth deviation d_i and depth score c_i^{depth} using Equation (4)
6:     Compute total scalar cost C_i = \alpha_{vlm} c_i^{vlm} + \alpha_{area} c_i^{area} 
                                      + \alpha_{conf} c_i^{conf} + \alpha_{depth} c_i^{depth}
7: End For
8: Compute \mathcal{C}_{risk}(m, n) = \max_i \mathcal{C}_i(m, n) for all pixels (m, n) \in \mathcal{M}_i
9: Return \mathcal{C}_{risk}
--------------------------------------------------------------------------------