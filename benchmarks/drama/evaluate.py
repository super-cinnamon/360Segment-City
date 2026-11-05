
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, brier_score_loss
from typing import List, Dict, Any

def bootstrap_ci(data, metric_fn, n_bootstrap=1000):
    """Computes 95% bootstrap confidence interval for a metric."""
    scores = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = np.random.choice(data, size=len(data), replace=True)
        scores.append(metric_fn(resampled))
    return np.percentile(scores, [2.5, 97.5])

def calculate_metrics(df):
    """
    Calculates risk detection metrics.
    """
    y_true = df["risk_label"].values
    y_score = df["risk_score"].values

    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_score)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    # F1 at best threshold
    best_f1 = 0
    best_thresh = 0
    thresholds = np.linspace(0, 5, 51)
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_thresh": best_thresh
    }

def evaluate_localization(df):
    """
    Object-level localization metrics.
    Matches hazard boxes to DRAMA risk boxes at IoU >= 0.5.
    """
    def compute_iou(box1, box2):
        if box1 is None or box2 is None: return 0
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

    hits = 0
    total_risk_clips = 0

    for _, row in df.iterrows():
        if row["risk_label"] == 1:
            total_risk_clips += 1
            # Top-1 hazard
            hazard_box = row["top_hazard_box"]
            gold_box = row["gold_risk_box"]
            if compute_iou(hazard_box, gold_box) >= 0.5:
                hits += 1

    return hits / total_risk_clips if total_risk_clips > 0 else 0

def main():
    # Load results
    with open("outputs/drama_results/all_results.jsonl", "r") as f:
        results = [json.loads(line) for line in f]

    # Load annotations for labels
    with open("dataset_root/integrated_output_v2.json", "r") as f:
        annos = json.load(f)

    # Merge into a dataframe
    data = []
    for r in results:
        cid = r["clip_id"]
        anno = next((a for a in annos if a["clip_id"] == cid), None)
        if not anno: continue

        data.append({
            "clip_id": cid,
            "risk_label": anno["risk_label"],
            "gold_risk_box": anno.get("bbox"), # Simplified: assume one gold box
            "risk_score": r.get("general_risk_score", 0),
            "top_hazard_box": r.get("top_hazard_box"),
        })

    df = pd.DataFrame(data)

    # 1. Risk Detection
    metrics = calculate_metrics(df)
    print(f"Risk Detection: {metrics}")

    # 2. Localization
    loc_hit_rate = evaluate_localization(df)
    print(f"Localization Hit Rate (IoU >= 0.5): {loc_hit_rate:.2%}")

if __name__ == "__main__":
    main()
