import os
import math
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONSTANTS
#
#   PATCH_SIZE : Must match the backbone's patch size (DINOv3 = 16 pixels).
#                Used to round image dimensions up to a multiple of 16 at
#                evaluation time so that patches tile perfectly.
#
#   SHORT_SIDE_MIN / SHORT_SIDE_MAX:
#                The shortest side of the image is uniformly sampled from
#                this range during random resizing (training stage 1).
#                Paper: "shortest side is uniformly sampled between 920
#                and the base resolution of the stage (1536 or 2048)."
#
#   EVAL_SHORT_SIDE:
#                At evaluation/inference time, "images are resized so that
#                the shortest side is 2048" (paper, Image Pre-Processing).
# ---------------------------------------------------------------------------
PATCH_SIZE      = 16
SHORT_SIDE_MIN  = 480     # reduced from 920 to fit below the new 800px MAX
SHORT_SIDE_MAX  = 800     # drastically reduced from 1536
EVAL_SHORT_SIDE = 800     # drastically reduced from 1024


# ---------------------------------------------------------------------------
# AUGMENTATION HELPERS
# ---------------------------------------------------------------------------

def _round_up_to_patch(size: int, patch: int = PATCH_SIZE) -> int:
    """
    Round an image dimension UP to the nearest multiple of `patch`.

    This ensures that every pixel row/column belongs to a complete patch,
    which is required by the ViT backbone.  For example, with patch_size=16:
        1537 → 1536  would leave an incomplete patch, so we go to 1552.
    """
    return math.ceil(size / patch) * patch


def _resize_to_short_side(img: Image.Image, short_side: int) -> Image.Image:
    """
    Resize a PIL image so its shortest side equals `short_side`,
    maintaining the original aspect ratio.
    """
    w, h = img.size
    if h < w:
        # Height is the shorter dimension.
        new_h = short_side
        new_w = int(w * short_side / h)
    else:
        # Width is the shorter (or equal) dimension.
        new_w = short_side
        new_h = int(h * short_side / w)
    return img.resize((new_w, new_h), Image.BILINEAR)


def train_transforms(img: Image.Image, target: dict) -> tuple:
    """
    Apply the training augmentation pipeline described in the paper.

    Two operations are applied in sequence:

    1. Random Horizontal Flip (p=0.5):
         If flipped, box x-coordinates are mirrored as well.

    2. Random Short-Side Resize — one of two branches chosen uniformly:
         (a) Direct random resize: shortest side ∈ [SHORT_SIDE_MIN, SHORT_SIDE_MAX].
         (b) Random crop (60–100% of area) followed by random resize as in (a).

    Args:
        img    : PIL Image (RGB).
        target : dict with 'boxes' (Tensor [N,4] in cxcywh normalised format)
                 and 'labels' (Tensor [N]).

    Returns:
        (img_tensor, target) where img_tensor is shape (3, H, W).
    """
    boxes  = target["boxes"].clone()   # (N, 4) [cx, cy, w, h], normalised
    labels = target["labels"].clone()  # (N,)

    W_orig, H_orig = img.size         # original pixel dimensions

    # ------------------------------------------------------------------
    # Step 1: Random Horizontal Flip (p=0.5)
    # ------------------------------------------------------------------
    if random.random() < 0.5:
        img = TF.hflip(img)
        # Mirror the x-coordinates of all boxes.
        # For [cx, cy, w, h] format: new_cx = 1.0 - cx
        # (cy, w, h are unaffected by horizontal flipping)
        if len(boxes) > 0:
            boxes[:, 0] = 1.0 - boxes[:, 0]

    # ------------------------------------------------------------------
    # Step 2: Random Resize Branch
    #
    #   Choose branch (a) or (b) with equal probability.
    # ------------------------------------------------------------------
    short_side = random.randint(SHORT_SIDE_MIN, SHORT_SIDE_MAX)

    if random.random() < 0.5:
        # ----------------------------------------------------------
        # Branch (a): Direct random resize.
        # ----------------------------------------------------------
        img = _resize_to_short_side(img, short_side)
        # Boxes are expressed in normalised coordinates → no change needed.

    else:
        # ----------------------------------------------------------
        # Branch (b): Random crop retaining 60–100% of the image area,
        #             then resize as in (a).
        #
        #   We compute a crop rectangle, apply it to the image, and
        #   update normalised box coordinates accordingly.
        # ----------------------------------------------------------
        # Sample a crop area as a fraction of the original image area.
        area_frac = random.uniform(0.6, 1.0)

        # Aspect ratio of the crop is sampled around the original ratio.
        for _ in range(10):  # Try up to 10 times for a valid crop.
            crop_area = area_frac * W_orig * H_orig
            # Randomly choose crop aspect ratio close to original.
            aspect = W_orig / H_orig
            crop_h = math.sqrt(crop_area / aspect)
            crop_w = math.sqrt(crop_area * aspect)
            crop_h = int(min(crop_h, H_orig))
            crop_w = int(min(crop_w, W_orig))

            if crop_h > 0 and crop_w > 0:
                # Random top-left corner for the crop.
                top  = random.randint(0, H_orig - crop_h)
                left = random.randint(0, W_orig - crop_w)
                img  = TF.crop(img, top, left, crop_h, crop_w)

                # Update normalised box coordinates to the new cropped space.
                # Original boxes are [cx, cy, w, h] normalised to [0,1] in
                # the original image. After cropping, we renormalise to the crop.
                if len(boxes) > 0:
                    # Convert normalised → pixel in original image.
                    cx_px = boxes[:, 0] * W_orig
                    cy_px = boxes[:, 1] * H_orig
                    w_px  = boxes[:, 2] * W_orig
                    h_px  = boxes[:, 3] * H_orig

                    # Convert to xyxy pixel coords.
                    x1 = cx_px - 0.5 * w_px - left
                    y1 = cy_px - 0.5 * h_px - top
                    x2 = cx_px + 0.5 * w_px - left
                    y2 = cy_px + 0.5 * h_px - top

                    # Clip to crop boundaries.
                    x1 = x1.clamp(0, crop_w)
                    y1 = y1.clamp(0, crop_h)
                    x2 = x2.clamp(0, crop_w)
                    y2 = y2.clamp(0, crop_h)

                    # Keep only boxes that still have positive area after crop.
                    keep = (x2 > x1) & (y2 > y1)
                    x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
                    labels = labels[keep]

                    # Convert back to [cx, cy, w, h] normalised to crop size.
                    cx_new = ((x1 + x2) / 2) / crop_w
                    cy_new = ((y1 + y2) / 2) / crop_h
                    w_new  = (x2 - x1) / crop_w
                    h_new  = (y2 - y1) / crop_h
                    boxes  = torch.stack([cx_new, cy_new, w_new, h_new], dim=-1)

                break  # Successful crop — exit retry loop.

        # Apply random resize to the cropped image.
        img = _resize_to_short_side(img, short_side)

    # ------------------------------------------------------------------
    # Step 3: Convert PIL image to float Tensor and normalise pixel values.
    #
    #   ImageNet mean & std are standard for DINOv2/v3 models.
    # ------------------------------------------------------------------
    img_tensor = TF.to_tensor(img)                          # [3, H, W], uint8 → float [0,1]
    img_tensor = TF.normalize(
        img_tensor,
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
    )

    target["boxes"]  = boxes
    target["labels"] = labels

    return img_tensor, target


def eval_transforms(img: Image.Image, target: dict) -> tuple:
    """
    Deterministic evaluation/inference transforms.

    Following the paper: "At evaluation time, images are resized so that
    the shortest side is 2048 without additional augmentation, and both
    sides are rounded up to the nearest multiple of the patch size."

    Args:
        img    : PIL Image (RGB).
        target : dict with 'boxes' and 'labels' (unchanged by eval transforms
                 since boxes are already in normalised coordinates).

    Returns:
        (img_tensor, target)
    """
    # Resize so the shortest side equals EVAL_SHORT_SIDE (2048).
    img = _resize_to_short_side(img, EVAL_SHORT_SIDE)

    # Round both dimensions up to the nearest multiple of PATCH_SIZE (16).
    # This ensures no incomplete patches at the image boundary when the
    # resized dimension is not already divisible by 16.
    w, h = img.size
    new_w = _round_up_to_patch(w)
    new_h = _round_up_to_patch(h)
    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.BILINEAR)

    # Convert and normalise.
    img_tensor = TF.to_tensor(img)
    img_tensor = TF.normalize(
        img_tensor,
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
    )

    # Boxes are normalised → no coordinate updates needed for pure resize.
    return img_tensor, target


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

class CustomDetectionDataset(Dataset):
    """
    A generic object detection dataset that wraps pre-loaded image paths
    and annotation dicts.

    Args:
        img_paths   : List of absolute paths to image files.
        annotations : List of dicts, each with:
                        'boxes'  : Tensor [N, 4], [cx, cy, w, h] normalised.
                        'labels' : Tensor [N], int64 class ids.
        mode        : 'train' applies training augmentations;
                      'eval'  applies evaluation transforms.
        percentage  : Float in (0, 1] — use only this fraction of the
                      dataset (useful for fast experimentation).
    """

    def __init__(
        self,
        img_paths:   list,
        annotations: list,
        mode:        str   = "train",   # "train" or "eval"
        percentage:  float = 1.0,
        seed:        int   = 42,        # Fixed seed for reproducible sampling
    ):
        # Package image paths and annotations together so they stay in sync.
        data = list(zip(img_paths, annotations))

        # Randomly shuffle the data before slicing to ensure the sub-sample
        # is a representative cross-section of the full dataset.
        if percentage < 1.0:
            random.Random(seed).shuffle(data)

        # Apply the percentage sub-sampling.
        sample_size = int(len(data) * percentage)
        self.img_paths   = [d[0] for d in data[:sample_size]]
        self.annotations = [d[1] for d in data[:sample_size]]
        self.mode        = mode

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image in RGB mode.
        img = Image.open(self.img_paths[idx]).convert("RGB")

        # Deep copy the target to avoid mutating the original annotation list.
        target = {
            "boxes":  self.annotations[idx]["boxes"].clone(),
            "labels": self.annotations[idx]["labels"].clone(),
        }

        # Apply the appropriate transform pipeline based on mode.
        if self.mode == "train":
            img_tensor, target = train_transforms(img, target)
        else:
            img_tensor, target = eval_transforms(img, target)

        # Return:
        #   img_tensor: Tensor [3, H, W] — float, normalised.
        #   target:     dict with 'boxes' [N, 4] and 'labels' [N].
        return img_tensor, target


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------
# BATCH PADDING HELPER
# ---------------------------------------------------------------------------

def pad_batch(images: list, patch_size: int = PATCH_SIZE, grid_size: int = 3):
    """
    Pad a batch of images to the maximum height and width in the batch.
    The final dimensions will be rounded up to the nearest multiple of
    (patch_size * grid_size) to ensure clean 3x3 windowing.
    """
    # Find the maximum height and width in the current batch.
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    # Round up to the nearest multiple of 48 (16 patch * 3 grid).
    stride = patch_size * grid_size
    padded_h = math.ceil(max_h / stride) * stride
    padded_w = math.ceil(max_w / stride) * stride

    # Create a batch tensor filled with zeros (padding).
    batch_shape = (len(images), 3, padded_h, padded_w)
    batched_imgs = images[0].new_full(batch_shape, 0.0)

    # Copy each image into the top-left corner of the padded tensor.
    for i, img in enumerate(images):
        h, w = img.shape[1:]
        batched_imgs[i, :, :h, :w].copy_(img)

    return batched_imgs


# ---------------------------------------------------------------------------

def train_plain_detr(
    model,
    train_loader:       DataLoader,
    criterion:          torch.nn.Module,
    optimizer:          torch.optim.Optimizer,
    device:             torch.device,
    epochs:             int   = 22,          # Stage 1 default from the paper
    save_path:          str   = "./checkpoints",
    start_epoch:        int   = 0,
    checkpoint_interval:int   = 1,           # Save checkpoint every N epochs
    val_loader:         DataLoader = None,
    scheduler=None,
    rank:               int   = 0,
    world_size:         int   = 1,
):
    """
    Main training loop for PlainDETR.

    Each epoch:
      1. Runs all training batches, calling forward + loss + backward.
      2. Optionally evaluates on the validation set.
      3. Saves a checkpoint to disk.

    Args:
        model               : PlainDETRFramework instance.
        train_loader        : DataLoader yielding (images_list, targets_list).
        criterion           : DualMatchingDETRCriterion instance.
        optimizer           : AdamW optimiser.
        device              : 'cuda' or 'cpu'.
        epochs              : Total number of training epochs.
        save_path           : Directory to save checkpoints.
        start_epoch         : Epoch to resume from (0 = fresh start).
        checkpoint_interval : How often (in epochs) to save checkpoints.
        val_loader          : Optional validation DataLoader.
        scheduler           : Optional LR scheduler (called after each epoch).
    """
    model.to(device)

    # ------------------------------------------------------------------
    # CHECKPOINT RESUME
    #
    #   If a "latest_checkpoint.pth" exists in save_path, load it to
    #   resume training from the last saved state.
    # ------------------------------------------------------------------
    checkpoint_file = os.path.join(save_path, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_file):
        if rank == 0:
            print(f"[Resume] Loading checkpoint: {checkpoint_file}")
        # map_location ensures parameters are loaded to the correct device
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if rank == 0:
            print(f"[Resume] Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):

        # In distributed mode, set_epoch ensures different shuffling each epoch
        if world_size > 1 and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # ------------------------------------------------------------------
        # TRAINING PHASE
        # ------------------------------------------------------------------
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch + 1}/{epochs} [train]",
            disable=(rank != 0)
        )

        for images, targets in pbar:
            # images is a list of Tensors with different H×W due to
            # random resizing. We pad them to a common size before stacking.
            images  = pad_batch(images).to(device)   # (B, 3, H, W)
            targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets
            ]

            optimizer.zero_grad()

            # --- Forward pass ---
            # outputs contains O2O and O2M predictions.
            outputs = model(images)

            # --- Loss computation ---
            # criterion returns a dict: {loss_focal, loss_bbox, loss_giou}
            # Each combines the O2O and O2M branch losses already summed.
            loss_dict   = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            # Weighted sum of all losses (e.g. 2*focal + 1*bbox + 2*giou).
            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict
                if k in weight_dict
            )

            # --- Backward pass and parameter update ---
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            pbar.set_postfix({
                "loss":      f"{losses.item():.4f}",
                "focal":     f"{loss_dict['loss_focal'].item():.4f}",
                "bbox":      f"{loss_dict['loss_bbox'].item():.4f}",
                "giou":      f"{loss_dict['loss_giou'].item():.4f}",
            })

        avg_loss = epoch_loss / max(len(train_loader), 1)
        if rank == 0:
            print(f"Epoch {epoch + 1} | Avg Train Loss: {avg_loss:.4f}")

        # Step LR scheduler once per epoch (if provided).
        if scheduler is not None:
            scheduler.step()

        # ------------------------------------------------------------------
        # VALIDATION PHASE (optional)
        # ------------------------------------------------------------------
        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            val_pbar = tqdm(
                val_loader, 
                desc=f"Epoch {epoch + 1}/{epochs} [val]",
                disable=(rank != 0)
            )
            with torch.no_grad():
                for val_images, val_targets in val_pbar:
                    val_images  = pad_batch(val_images).to(device)
                    val_targets = [
                        {k: v.to(device) for k, v in t.items()}
                        for t in val_targets
                    ]

                    val_outputs   = model(val_images)
                    val_loss_dict = criterion(val_outputs, val_targets)

                    val_losses = sum(
                        val_loss_dict[k] * weight_dict[k]
                        for k in val_loss_dict
                        if k in weight_dict
                    )
                    val_loss += val_losses.item()
                    val_pbar.set_postfix({"val_loss": f"{val_losses.item():.4f}"})

            avg_val_loss = val_loss / max(len(val_loader), 1)
            if rank == 0:
                print(f"Epoch {epoch + 1} | Avg Val Loss: {avg_val_loss:.4f}")

        # ------------------------------------------------------------------
        # CHECKPOINT SAVING
        # ------------------------------------------------------------------
        if (epoch + 1) % checkpoint_interval == 0 and rank == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            state = {
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":                avg_loss,
            }

            # Save a named checkpoint for this specific epoch.
            torch.save(state, os.path.join(save_path, f"checkpoint_epoch_{epoch + 1}.pth"))
            # Also overwrite the "latest" checkpoint (used for resuming).
            torch.save(state, checkpoint_file)
            print(f"[Checkpoint] Saved to {save_path} (epoch {epoch + 1})")

    print("Training finished!")


# ---------------------------------------------------------------------------
# CHECKPOINT SAVE UTILITY (standalone, for external use)
# ---------------------------------------------------------------------------
def save_checkpoint(model, path, epoch):
    """
    Save only the model's state_dict to a file.
    Useful for saving inference-only snapshots.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, f"plain_detr_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[Checkpoint] Model state saved to {save_path}")
