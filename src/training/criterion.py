import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# LOSS WEIGHTS (from the paper, Section "Training")
#
#   loss_focal : weight = 2.0  (Focal Loss on class logits)
#   loss_bbox  : weight = 1.0  (L1 loss on normalised box coordinates)
#   loss_giou  : weight = 2.0  (Generalised IoU loss, default stages 1-2;
#                               set to 4.0 at construction time for stage 3)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HELPERS: BOX CONVERSION & GENERALISED IoU
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(x):
    """
    Convert bounding boxes from centre-format [cx, cy, w, h]
    to corner-format [x1, y1, x2, y2].

    This is needed because GIoU requires corner coordinates, while the
    model outputs—and the targets—use the normalised [cx, cy, w, h] format.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w),   # x1
        (y_c - 0.5 * h),   # y1
        (x_c + 0.5 * w),   # x2
        (y_c + 0.5 * h),   # y2
    ]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalised Intersection-over-Union (GIoU) matrix.
    Reference: Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019.

    Args:
        boxes1 : Tensor (N, 4) in xyxy format.
        boxes2 : Tensor (M, 4) in xyxy format.

    Returns:
        Tensor (N, M) of pairwise GIoU values in [-1, 1].
        Higher is better (=1 means perfect overlap).

    GIoU extends standard IoU by penalising predictions that don't
    overlap with any ground-truth box, which gives gradients even for
    non-overlapping boxes and improves convergence.
    """
    # Per-box areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

    # Intersection rectangle (top-left corner = max of top-lefts, etc.)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # (N, M, 2)

    # Clamp to 0 so non-overlapping boxes give 0 intersection area.
    wh    = (rb - lt).clamp(min=0)                        # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]                    # (N, M)

    union = area1[:, None] + area2 - inter                # (N, M)
    iou   = inter / union.clamp(min=1e-6)                 # (N, M)

    # Smallest enclosing box (used to compute the GIoU penalty term).
    lti   = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rbi   = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    whi   = (rbi - lti).clamp(min=0)                       # (N, M, 2)
    areai = whi[:, :, 0] * whi[:, :, 1]                    # (N, M)

    # GIoU = IoU  −  (area of enclosing box − union) / area of enclosing box
    giou  = iou - (areai - union) / areai.clamp(min=1e-6)  # (N, M)
    return giou


# ---------------------------------------------------------------------------
# FOCAL LOSS
#
#   Following the paper: "We use the Focal Loss (Lin et al., 2018) as
#   classification loss, with a weight of 2."
#
#   Focal Loss down-weights well-classified easy examples and forces the
#   model to focus on hard, misclassified ones.  The formulation is:
#
#     FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
#
#   where p_t is the predicted probability for the ground-truth class.
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Sigmoid-based Focal Loss.

    Unlike softmax cross-entropy, this applies sigmoid independently to
    each logit, treating the classification head as C binary classifiers.
    This is the standard choice for DETR-family detectors (DINO, DAB-DETR,
    DN-DETR, etc.).

    Args:
        inputs  : Raw logits, shape (N, num_classes).
        targets : One-hot encoded float labels, shape (N, num_classes).
        alpha   : Balancing factor for positive/negative examples (0.25).
        gamma   : Focusing exponent (2.0 from the Lin et al. paper).
        reduction: "mean" | "sum" | "none".

    Returns:
        Scalar focal loss (or per-element if reduction="none").
    """
    # Sigmoid to convert logits → probabilities per class.
    p   = torch.sigmoid(inputs)

    # Binary cross-entropy per element (no reduction yet).
    # ce_loss[i,c] = -log(p[i,c]) if targets[i,c]==1,
    #              = -log(1-p[i,c]) if targets[i,c]==0
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    # p_t is the predicted probability for the target class, i.e.
    #   p   where targets == 1
    #   1-p where targets == 0
    p_t = p * targets + (1 - p) * (1 - targets)

    # Focal weight: (1 - p_t)^gamma — down-weights easy examples.
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting: alpha for positives, (1-alpha) for negatives.
    alpha_t      = alpha * targets + (1 - alpha) * (1 - targets)

    # Final focal loss per element.
    focal_loss   = alpha_t * focal_weight * ce_loss

    if reduction == "mean":
        return focal_loss.mean()
    elif reduction == "sum":
        return focal_loss.sum()
    return focal_loss


# ---------------------------------------------------------------------------
# DUAL-MATCHING DETR CRITERION
#
#   This criterion implements the two-branch matching strategy:
#
#     Branch 1 — One-to-One (O2O):
#       Standard Hungarian bipartite matching.  Each predicted query is
#       matched to AT MOST one ground-truth box.  The matched pairs form
#       the primary detection-level supervision signal.
#
#     Branch 2 — One-to-Many (O2M):
#       The GT set is replicated K times (K copies per GT object), and
#       Hungarian matching is applied on this expanded target set.  This
#       allows multiple queries to supervise the same object, encouraging
#       better recall and faster convergence (Group-DETR / H-DETR style).
#
#   Both branches share the same prediction heads (class + bbox MLP)
#   but receive predictions from two different sets of query embeddings
#   (o2o and o2m, respectively).  The losses from both branches are
#   added together before back-propagation.
# ---------------------------------------------------------------------------

class DualMatchingDETRCriterion(nn.Module):
    """
    Dual-matching DETR loss combining:
      - Focal Loss  (weight = 2)
      - L1 bbox loss (weight = 1)
      - GIoU loss   (weight = 2, or configurable)

    Applied independently to the O2O and O2M prediction sets and summed.
    """

    def __init__(
        self,
        num_classes: int,
        focal_alpha: float = 0.25,     # Focal Loss alpha
        focal_gamma: float = 2.0,      # Focal Loss gamma
        o2m_k: int = 5,               # Number of GT copies for O2M branch
        giou_weight: float = 2.0,     # Set to 4.0 for training stage 3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.o2m_k       = o2m_k       # How many copies of GT to make for O2M

        # The weight_dict maps loss keys → their scalar multipliers.
        # It is accessed by the training loop to compute the weighted sum.
        self.weight_dict = {
            "loss_focal": 2.0,         # weight=2 from the paper
            "loss_bbox":  1.0,         # weight=1 from the paper
            "loss_giou":  giou_weight, # weight=2 (or 4 in stage 3)
        }

    # -----------------------------------------------------------------------
    # _get_src_permutation_idx
    #
    #   Given a list of (row_indices, col_indices) tuples from the Hungarian
    #   solver (one per image in the batch), this helper creates a pair of
    #   index tensors that can be used to gather the matched predicted boxes
    #   and logits from the batch-level prediction tensors.
    # -----------------------------------------------------------------------
    def _get_src_permutation_idx(self, indices):
        """
        Returns (batch_idx, src_idx) — two 1-D tensors — where
        batch_idx[k] and src_idx[k] together identify the k-th matched
        prediction across the whole batch.
        """
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    # -----------------------------------------------------------------------
    # _compute_losses_for_branch
    #
    #   Core loss computation for ONE set of predictions (either O2O or O2M).
    #   Steps:
    #     1. Build a cost matrix (class + bbox + GIoU) across all queries
    #        vs all targets (in the batch).
    #     2. Solve bipartite matching with scipy's linear_sum_assignment.
    #     3. Compute Focal Loss, L1 loss, and GIoU loss for matched pairs.
    # -----------------------------------------------------------------------
    def _compute_losses_for_branch(
        self,
        pred_logits: torch.Tensor,   # (B, Q, num_classes+1)
        pred_boxes:  torch.Tensor,   # (B, Q, 4)
        targets:     list,           # list of B dicts with 'boxes' and 'labels'
    ):
        """
        Compute losses for a single prediction branch (O2O or O2M).

        Returns:
            dict with keys 'loss_focal', 'loss_bbox', 'loss_giou'.
        """
        device = pred_logits.device
        B, num_queries, _ = pred_logits.shape

        # ------------------------------------------------------------------
        # STEP 1: Flatten batch × queries for cost matrix computation.
        # ------------------------------------------------------------------
        # out_prob: (B*Q, num_classes+1) — sigmoid probabilities per class
        out_prob = torch.sigmoid(pred_logits.flatten(0, 1))
        # out_bbox: (B*Q, 4) — normalised [cx, cy, w, h]
        out_bbox = pred_boxes.flatten(0, 1)

        # Concatenate all GT labels and boxes across the batch.
        tgt_ids  = torch.cat([v["labels"] for v in targets])   # (T_total,)
        tgt_bbox = torch.cat([v["boxes"]  for v in targets])   # (T_total, 4)

        # Initialise losses to zero (with grad to avoid graph disconnection).
        loss_focal = torch.tensor(0.0, device=device)
        loss_bbox  = torch.tensor(0.0, device=device)
        loss_giou  = torch.tensor(0.0, device=device)

        if len(tgt_ids) == 0:
            # No objects in this batch — attach to predictions to keep graph.
            loss_focal = pred_logits.sum() * 0.0
            loss_bbox  = pred_boxes.sum()  * 0.0
            loss_giou  = pred_boxes.sum()  * 0.0
            return {
                "loss_focal": loss_focal,
                "loss_bbox":  loss_bbox,
                "loss_giou":  loss_giou,
            }

        # ------------------------------------------------------------------
        # STEP 2: BIPARTITE MATCHING (Hungarian algorithm)
        #
        #   We compute a cost matrix C of shape (B*Q, T_total), where each
        #   entry measures how expensive it is to match predicted query i
        #   with ground-truth target j.  We then call linear_sum_assignment
        #   independently per image (splitting by GT counts).
        #
        #   Cost components (following the paper's loss weights):
        #     cost_class : Classification cost — negative predicted prob
        #                  for the GT class.  Using probability (not FL)
        #                  here, since cost matrices should be comparable
        #                  in scale across elements.
        #     cost_bbox  : L1 distance between predicted and GT boxes.
        #     cost_giou  : Negative GIoU between predicted and GT boxes.
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Classification cost: -(prob of the GT class for each query)
            # Shape: (B*Q, T_total)
            cost_class = -out_prob[:, tgt_ids]

            # L1 bounding box cost.
            # Shape: (B*Q, T_total)
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # GIoU cost (negative, because lower cost = better match).
            # Convert boxes to xyxy for GIoU computation.
            out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)     # (B*Q, 4)
            tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)     # (T_total, 4)
            cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)  # (B*Q, T_total)

            # Combine into a single weighted cost matrix.
            # Using the same weights as the training losses (from paper).
            C = (
                self.weight_dict["loss_focal"] * cost_class
                + self.weight_dict["loss_bbox"] * cost_bbox
                + self.weight_dict["loss_giou"] * cost_giou
            )   # (B*Q, T_total)

            # Reshape to (B, Q, T_total) so we can split by image.
            C = C.view(B, num_queries, -1).cpu()

            # sizes[i] = number of GT objects in image i of the batch.
            sizes = [len(v["boxes"]) for v in targets]

            # Run Hungarian matching independently per image.
            # C.split(sizes, -1) gives B tensors, each (B, Q, T_i).
            # We index C[i] to get (Q, T_i) for image i.
            indices = [
                linear_sum_assignment(c[i])
                for i, c in enumerate(C.split(sizes, -1))
            ]
            # Convert numpy arrays back to tensors.
            indices = [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in indices
            ]

        # ------------------------------------------------------------------
        # STEP 3: COMPUTE LOSSES BASED ON MATCHED PAIRS
        # ------------------------------------------------------------------
        # idx is a tuple (batch_idx, src_idx) mapping each matched query
        # to its batch image and its position in the query sequence.
        idx = self._get_src_permutation_idx(indices)

        # Number of matched targets across the whole batch (for normalisation).
        num_targets = sum(len(v["labels"]) for v in targets)
        num_targets = max(num_targets, 1)  # avoid division by zero

        # --- FOCAL LOSS ---
        # Build one-hot target tensors for the focal loss:
        #   target_classes[b, q] = GT class id if query (b,q) is matched,
        #                        = num_classes (background) otherwise.
        target_classes = torch.full(
            pred_logits.shape[:2],       # (B, Q)
            self.num_classes,            # fill with background index
            dtype=torch.int64,
            device=device,
        )
        # Fill in the actual class ids at matched positions.
        target_classes[idx] = tgt_ids

        # Convert integer class ids to one-hot float tensors for sigmoid FL.
        # Shape: (B, Q, num_classes + 1)
        target_classes_onehot = torch.zeros(
            [B, num_queries, self.num_classes + 1],
            dtype=torch.float32,
            device=device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # Drop the background column — we compute FL only over foreground.
        # Shape: (B, Q, num_classes)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_focal = sigmoid_focal_loss(
            pred_logits[:, :, :-1],      # (B, Q, num_classes) — drop BG logit
            target_classes_onehot,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="sum",
        ) / num_targets                  # normalise by number of GT objects

        # --- L1 BOUNDING BOX LOSS ---
        # Only compute box losses for the matched predictions.
        src_boxes    = pred_boxes[idx]   # (num_targets, 4)
        target_boxes = tgt_bbox          # (num_targets, 4), already concatenated

        loss_bbox = F.l1_loss(
            src_boxes, target_boxes, reduction="none"
        ).sum() / num_targets

        # --- GIoU LOSS ---
        # GIoU between each matched (src_box, target_box) pair.
        # We want GIoU → 1  (perfect overlap), so loss = 1 − GIoU.
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss_giou = loss_giou.sum() / num_targets

        return {
            "loss_focal": loss_focal,
            "loss_bbox":  loss_bbox,
            "loss_giou":  loss_giou,
        }

    # -----------------------------------------------------------------------
    # FORWARD
    #
    #   Entry point called by the training loop.
    #
    #   Args:
    #     outputs : dict returned by PlainDETRFramework.forward(), containing:
    #                 "pred_logits"     (B, 1500, C+1)  — O2O class logits
    #                 "pred_boxes"      (B, 1500, 4)    — O2O box predictions
    #                 "pred_logits_o2m" (B, 1500, C+1)  — O2M class logits
    #                 "pred_boxes_o2m"  (B, 1500, 4)    — O2M box predictions
    #     targets : list of B dicts, each containing:
    #                 "boxes"  : (N_i, 4) GT boxes in [cx, cy, w, h] normalised
    #                 "labels" : (N_i,)   GT class ids (int64)
    #
    #   Returns:
    #     dict whose keys match self.weight_dict, with scalar tensor values.
    # -----------------------------------------------------------------------
    def forward(self, outputs, targets):
        # ------------------------------------------------------------------
        # BRANCH 1 — ONE-TO-ONE LOSS
        #
        #   Standard Hungarian matching on the O2O prediction set.
        #   Each GT object is matched to at most one query.
        # ------------------------------------------------------------------
        losses_o2o = self._compute_losses_for_branch(
            pred_logits=outputs["pred_logits"],
            pred_boxes= outputs["pred_boxes"],
            targets=targets,
        )

        # ------------------------------------------------------------------
        # BRANCH 2 — ONE-TO-MANY LOSS
        #
        #   We replicate the GT set self.o2m_k times so that the O2M set
        #   of 1500 queries can share supervision of the same objects.
        #   This accelerates convergence and improves recall.
        #
        #   Specifically, for each image we replace its target dict with
        #   a new dict whose 'boxes' and 'labels' are repeated K times.
        # ------------------------------------------------------------------
        # Build the expanded target list (K copies of each GT object).
        targets_o2m = [
            {
                "boxes":  t["boxes"].repeat(self.o2m_k, 1),    # (K*N_i, 4)
                "labels": t["labels"].repeat(self.o2m_k),      # (K*N_i,)
            }
            for t in targets
        ]

        losses_o2m = self._compute_losses_for_branch(
            pred_logits=outputs["pred_logits_o2m"],
            pred_boxes= outputs["pred_boxes_o2m"],
            targets=targets_o2m,
        )

        # ------------------------------------------------------------------
        # COMBINE LOSSES
        #
        #   Sum the O2O and O2M losses for each loss type.  The training
        #   loop then multiplies each by its weight from self.weight_dict.
        # ------------------------------------------------------------------
        combined_losses = {
            "loss_focal": losses_o2o["loss_focal"] + losses_o2m["loss_focal"],
            "loss_bbox":  losses_o2o["loss_bbox"]  + losses_o2m["loss_bbox"],
            "loss_giou":  losses_o2o["loss_giou"]  + losses_o2m["loss_giou"],
        }

        return combined_losses
