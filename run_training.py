import os
import json
import glob
import math
import argparse

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------------------------------------
from src.config.utils import CONFIG
from src.training.train import CustomDetectionDataset, train_plain_detr
from src.training.dino_object_detection import PlainDETRFramework
from src.training.criterion import DualMatchingDETRCriterion

# ---------------------------------------------------------------------------
# BDD100K CLASS DEFINITIONS
#
#   The BDD100K dataset contains 10 detection categories.  We map each
#   string label to a 0-indexed integer ID used by the model's class head.
# ---------------------------------------------------------------------------
BDD100K_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign",
]
CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(BDD100K_CLASSES)}


# ---------------------------------------------------------------------------
# DATA LOADING
#
#   BDD100K stores one JSON label file per sequence (video), where the
#   "frames" array contains per-frame object annotations.  This function
#   iterates all label files, extracts the bounding boxes and class labels
#   for the corresponding image, and converts boxes to the model's expected
#   format: [cx, cy, w, h] normalised to [0, 1].
# ---------------------------------------------------------------------------
def load_bdd100k_data(image_dir: str, label_dir: str):
    """
    Load BDD100K images and annotations.

    Args:
        image_dir : Path to the directory containing BDD100K .jpg images.
        label_dir : Path to the directory containing BDD100K .json label files.

    Returns:
        img_paths   : List of absolute paths to images that have ≥1 valid box.
        annotations : Parallel list of dicts {'boxes': Tensor, 'labels': Tensor}.
    """
    img_paths   = []
    annotations = []

    # Glob all JSON files in the label directory.
    label_files = glob.glob(os.path.join(label_dir, "*.json"))

    for label_file in label_files:
        with open(label_file, "r") as f:
            data = json.load(f)

        # BDD100K label format: data["name"] = image basename (without extension).
        img_name = data["name"] + ".jpg"
        img_path = os.path.join(image_dir, img_name)

        # Skip if the corresponding image doesn't exist on disk.
        if not os.path.exists(img_path):
            continue

        boxes  = []
        labels = []

        # BDD100K original image resolution (used for normalisation).
        IMG_W = 1280.0
        IMG_H = 720.0

        # Each label file may contain multiple frames (video sequences).
        # We collect all 2-D bounding boxes across all frames.
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                cat = obj.get("category", "")

                # Skip objects whose category is not in our 10-class vocabulary
                # or that don't have a "box2d" annotation.
                if cat not in CLASS_NAME_TO_ID or "box2d" not in obj:
                    continue

                # Extract axis-aligned box in pixel coordinates.
                x1 = obj["box2d"]["x1"]
                y1 = obj["box2d"]["y1"]
                x2 = obj["box2d"]["x2"]
                y2 = obj["box2d"]["y2"]

                # Convert from [x1, y1, x2, y2] pixel coords to
                # [cx, cy, w, h] normalised to [0, 1] in image space.
                w  = (x2 - x1) / IMG_W
                h  = (y2 - y1) / IMG_H
                cx = (x1 / IMG_W) + (w / 2)
                cy = (y1 / IMG_H) + (h / 2)

                boxes.append([cx, cy, w, h])
                labels.append(CLASS_NAME_TO_ID[cat])

        # Only include images that have at least one annotated object.
        if len(boxes) > 0:
            img_paths.append(img_path)
            annotations.append({
                "boxes":  torch.tensor(boxes,  dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    return img_paths, annotations


# ---------------------------------------------------------------------------
# COLLATE FUNCTION
#
#   PyTorch's default collate_fn tries to stack all elements into batched
#   tensors, which fails when images have different sizes (due to variable-
#   resolution augmentation) or when target dicts have variable-length
#   box tensors.  This custom collate_fn keeps images and targets as lists.
# ---------------------------------------------------------------------------
def collate_fn(batch):
    """
    Args:
        batch : List of (img_tensor, target_dict) tuples from the Dataset.

    Returns:
        images  : List[Tensor(3, H_i, W_i)] — one tensor per image.
        targets : List[dict]                — one dict per image.
    """
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ---------------------------------------------------------------------------
# LEARNING RATE SCHEDULER — WARMUP + COSINE DECAY
#
#   As described in the paper (stage 3 specifically):
#   "After a linear warmup of 2000 iterations, the learning rate follows
#    a cosine decay schedule, starting at 2.5e-5 and reaching 2.5e-6
#    at the 8th epoch."
#
#   We implement a combined scheduler using PyTorch's SequentialLR:
#     Phase 1: LinearLR  — ramps from warmup_factor * base_lr up to base_lr
#              over `warmup_epochs` epochs.
#     Phase 2: CosineAnnealingLR — decays from base_lr to eta_min over
#              the remaining `cosine_epochs` epochs.
# ---------------------------------------------------------------------------
def build_lr_scheduler(
    optimizer:      torch.optim.Optimizer,
    warmup_epochs:  int,
    total_epochs:   int,
    warmup_factor:  float = 0.1,   # LR starts at warmup_factor * base_lr
    eta_min:        float = 2.5e-6, # Minimum LR after cosine decay
):
    """
    Build a SequentialLR scheduler: linear warmup then cosine decay.

    Args:
        optimizer     : The AdamW optimiser whose LR we are scheduling.
        warmup_epochs : Number of epochs for the linear warm-up phase.
        total_epochs  : Total training epochs (determines cosine period).
        warmup_factor : Multiplier for the initial (pre-warmup) LR.
        eta_min       : Minimum LR reached at the end of cosine decay.

    Returns:
        torch.optim.lr_scheduler.SequentialLR instance.
    """
    # Phase 1: Linear ramp from (warmup_factor * base_lr) → base_lr.
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Phase 2: Cosine decay from base_lr → eta_min over remaining epochs.
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=eta_min,
    )

    # SequentialLR switches schedulers after warmup_epochs milestones.
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return scheduler


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PlainDETR on BDD100K")

    # --- Data paths ---
    parser.add_argument("--image_dir",     type=str,   default="data/bdd/images/100k/train")
    parser.add_argument("--label_dir",     type=str,   default="data/bdd/labels/100k/train")
    parser.add_argument("--val_image_dir", type=str,   default="data/bdd/images/100k/val")
    parser.add_argument("--val_label_dir", type=str,   default="data/bdd/labels/100k/val")

    # --- Training hyper-parameters ---
    # The paper trains in 3 stages.  Default values match Stage 1:
    #   22 epochs, LR 5e-5, weight_decay 0.05.
    parser.add_argument("--epochs",              type=int,   default=22,
                        help="Total training epochs (Stage 1=22, Stage 2=4, Stage 3=12)")
    parser.add_argument("--batch_size",          type=int,   default=1,
                        help="Per-GPU batch size (paper uses 32 total over 32 GPUs)")
    parser.add_argument("--lr",                  type=float, default=5e-5,
                        help="Base learning rate (Stage 1=5e-5, Stage 2=2.5e-5, Stage 3=2.5e-5)")
    parser.add_argument("--weight_decay",        type=float, default=0.05,
                        help="AdamW weight decay (paper: 0.05)")
    parser.add_argument("--warmup_epochs",       type=int,   default=1,
                        help="Number of warm-up epochs before LR reaches base_lr")
    parser.add_argument("--giou_weight",         type=float, default=2.0,
                        help="GIoU loss weight (2.0 for stages 1-2, 4.0 for stage 3)")
    parser.add_argument("--o2m_k",              type=int,   default=5,
                        help="Number of GT copies for the one-to-many branch")

    # --- Infrastructure ---
    parser.add_argument("--checkpoint_interval", type=int,   default=1)
    parser.add_argument("--save_path",           type=str,   default="./checkpoints")
    parser.add_argument("--percentage",          type=float, default=1.0,
                        help="Fraction of training dataset to use (0–1)")
    parser.add_argument("--val_percentage",      type=float, default=None,
                        help="Fraction of validation dataset to use (0–1). If None, uses same as --percentage.")

    args = parser.parse_args()

    # If val_percentage isn't specified, fall back to the training percentage.
    if args.val_percentage is None:
        args.val_percentage = args.percentage

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    print("Loading training data...")
    img_paths, annotations = load_bdd100k_data(args.image_dir, args.label_dir)
    print(f"  → {len(img_paths)} training images with annotations.")

    print("Loading validation data...")
    val_img_paths, val_annotations = load_bdd100k_data(args.val_image_dir, args.val_label_dir)
    print(f"  → {len(val_img_paths)} validation images with annotations.")

    # ------------------------------------------------------------------
    # DATASETS & DATALOADERS
    #
    #   Training set: 'train' mode applies the full augmentation pipeline
    #                 (flip, random crop, random resize).
    #   Validation set: 'eval' mode applies only deterministic resize.
    # ------------------------------------------------------------------
    train_dataset = CustomDetectionDataset(
        img_paths, annotations,
        mode="train",
        percentage=args.percentage,
    )
    val_dataset = CustomDetectionDataset(
        val_img_paths, val_annotations,
        mode="eval",
        percentage=args.val_percentage,
    )

    # num_workers=0 keeps data loading in the main process; increase if you
    # have many CPU cores and I/O is a bottleneck.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,                  # Shuffle for better gradient diversity
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,               # Faster host→GPU transfers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,                 # Deterministic order for validation
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # MODEL
    #
    #   Backbone is loaded from CONFIG and kept frozen (as per the paper).
    # ------------------------------------------------------------------
    print("Initialising model...")
    model = PlainDETRFramework(
        model_name=CONFIG["train"]["dino_backbone"],
        num_classes=CONFIG["train"]["num_classes"],
        frozen_backbone=True,          # Backbone parameters are never updated
    )
    print(f"  Backbone    : {CONFIG['train']['dino_backbone']}")
    print(f"  Num classes : {CONFIG['train']['num_classes']}")

    # ------------------------------------------------------------------
    # LOSS CRITERION
    #
    #   DualMatchingDETRCriterion computes:
    #     1. O2O Hungarian-matched Focal + L1 + GIoU loss.
    #     2. O2M duplicate-target matched Focal + L1 + GIoU loss.
    #   Both are summed before returning.
    # ------------------------------------------------------------------
    criterion = DualMatchingDETRCriterion(
        num_classes=CONFIG["train"]["num_classes"],
        focal_alpha=0.25,              # Down-weighting factor for easy examples
        focal_gamma=2.0,              # Focusing exponent
        o2m_k=args.o2m_k,            # How many GT copies for the O2M branch
        giou_weight=args.giou_weight, # 2.0 for stages 1-2, 4.0 for stage 3
    )

    # ------------------------------------------------------------------
    # OPTIMIZER — AdamW
    #
    #   "Throughout training, we use the AdamW optimizer (Loshchilov and
    #    Hutter, 2017) with a weight decay of 0.05." (paper, Training)
    #
    #   We only pass parameters that require gradients (i.e. the encoder,
    #   decoder, projection layers, and heads — NOT the frozen backbone).
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,   # 0.05 from the paper
        betas=(0.9, 0.999),               # Standard AdamW betas
    )

    # ------------------------------------------------------------------
    # LR SCHEDULER — Linear Warmup + Cosine Decay
    #
    #   Paper Stage 1: "After an initial warmup of 1000 steps, the
    #   learning rate is set to 5e-5 and is divided by 10 after the
    #   20th epoch."
    #   We approximate the 1000-step warmup as `warmup_epochs` full epochs
    #   and implement the decay as a cosine schedule (a smooth analogue
    #   of the step decay described in the paper).
    # ------------------------------------------------------------------
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        warmup_factor=0.1,                # Start at 10% of base_lr
        eta_min=args.lr / 10,             # Decay to base_lr / 10 (≈ paper's ÷10)
    )

    # ------------------------------------------------------------------
    # DEVICE SELECTION
    #
    #   Use CUDA if available.  Single-GPU training only; multi-GPU
    #   (DDP) would require additional setup not included here.
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    print("\nStarting training loop...")
    train_plain_detr(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        save_path=args.save_path,
        checkpoint_interval=args.checkpoint_interval,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
