import os
import math
import dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.config.utils import CONFIG

# ---------------------------------------------------------------------------
# Load environment variables (e.g., HuggingFace token for gated models).
# ---------------------------------------------------------------------------
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------------------------------------------------------------------
# KEY ARCHITECTURAL CONSTANTS (from the paper)
#
#   PATCH_SIZE        : DINOv3 uses 16x16 pixel patches (ViT-7B/16).
#                       Each patch becomes one token in the backbone.
#   NUM_INTERMEDIATE  : We extract features from 4 intermediate backbone
#                       layers (relative depths: 25%, 50%, 75%, 100% of
#                       the total number of layers — corresponding to
#                       layers [10, 20, 30, 40] in a 40-layer ViT-7B).
#   ENCODER_DIM       : The transformer encoder & decoder both operate at
#                       768-dimensional embeddings.
#   ENCODER_LAYERS    : 6 self-attention blocks in the encoder.
#   DECODER_LAYERS    : 6 cross-attention blocks in the decoder.
#   NUM_QUERIES_O2O   : 1500 "one-to-one" queries — each matched to at
#                       most one ground truth object (standard DETR logic).
#   NUM_QUERIES_O2M   : 1500 "one-to-many" queries — each is allowed to
#                       match multiple copies of the target set during
#                       training (improves recall, à la Group-DETR / H-DETR).
#   NUM_WINDOWS       : The image is divided into a 3×3 grid of windows;
#                       each window is passed through the backbone independently.
# ---------------------------------------------------------------------------
PATCH_SIZE       = 16
NUM_INTERMEDIATE = 4
ENCODER_DIM      = 768
ENCODER_LAYERS   = 6
DECODER_LAYERS   = 6
NUM_QUERIES_O2O  = 1500
NUM_QUERIES_O2M  = 1500
NUM_WINDOWS_GRID = 3   # 3×3 = 9 windows total


class PlainDETRFramework(nn.Module):
    """
    PlainDETR-style object detection framework built on top of a frozen
    DINOv3 ViT backbone.

    Architecture summary (following the paper):

      1. BACKBONE  – A frozen DINOv3 ViT-7B/16.
         For every image, the backbone is called multiple times:
           a. Once per window tile in a 3×3 spatial grid.
           b. Once on a downscaled version of the whole image.
         In all cases, we extract features from 4 intermediate layers
         (at relative depths ≈ 25%, 50%, 75%, 100%) and concatenate
         them channel-wise, yielding a 4×H dimension per token.

      2. WINDOWING  – The spatial feature maps from (1a) are reassembled
         into one large map.  The global map from (1b) is bilinearly
         upsampled to the same spatial resolution and concatenated
         channel-wise, doubling the channel dimension to 8×H.

      3. ENCODER  – A stack of 6 Transformer self-attention blocks with
         d_model=768.  The windowed features are projected to 768 and
         then fed into the encoder to build a contextualised feature map.

      4. DECODER  – A stack of 6 Transformer cross-attention blocks with
         d_model=768.
         Two sets of object queries attend to the encoder output:
           • 1500 "one-to-one" (O2O) queries → matched via bipartite
             Hungarian matching (one prediction per ground-truth box).
           • 1500 "one-to-many" (O2M) queries → matched against k
             replicated copies of the target set to improve recall.
         An attention mask prevents the two groups from attending to
         each other during self-attention steps inside the decoder.

      5. HEADS  – Two shared MLP heads (class + bbox) applied to both
         groups of decoder outputs.
    """

    def __init__(
        self,
        model_name:      str  = CONFIG["train"]["dino_backbone"],
        num_classes:     int  = CONFIG["train"]["num_classes"],
        frozen_backbone: bool = True,
        token:           str  = HF_TOKEN,
        memory_efficient: bool = None,  # If None, auto-detects based on system RAM
    ):
        """
        Args:
            model_name       : HuggingFace model path for DINOv3 backbone.
            num_classes      : Number of object categories in the dataset.
            frozen_backbone  : If True, backbone weights are not updated.
            token            : HuggingFace access token (required for DINOv3).
            memory_efficient : If True, uses a sum-of-projections strategy to
                               save ~20x peak memory. If None, auto-detects.
            use_checkpoint   : If True, uses gradient checkpointing for the
                               transformer encoder to save activation memory.
        """
        super().__init__()

        # --- AUTO-DETECT MEMORY LIMIT ---
        if memory_efficient is None:
            try:
                import psutil
                # If total RAM is less than 32GB, default to memory efficient
                # to avoid allocation failures with large batches.
                total_ram_gb = psutil.virtual_memory().total / (1024**3)
                memory_efficient = (total_ram_gb < 32.0)
            except ImportError:
                # Default to safe mode if psutil is missing
                memory_efficient = True

        self.memory_efficient = memory_efficient
        self.use_checkpoint = False  # Disabled by default (caused hangs on some CPUs)

        # ------------------------------------------------------------------
        # 1. BACKBONE — DINOv3 ViT (loaded from HuggingFace).
        #
        #    We keep the backbone entirely frozen as described in the paper:
        #    "making it the first competitive detection model to do so."
        #    The backbone's hidden_size attribute gives us H (e.g. 4096
        #    for ViT-7B), which we need for downstream projections.
        # ------------------------------------------------------------------
        self.backbone = AutoModel.from_pretrained(model_name, token=token)
        self.embed_dim = self.backbone.config.hidden_size  # H per layer (e.g. 4096)

        if frozen_backbone:
            # Freeze every parameter in the backbone — no gradients flow
            # back through the vision encoder during training.
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------
        # 2. INTERMEDIATE LAYER INDICES
        #
        #    The paper extracts layers [10, 20, 30, 40] from a 40-layer
        #    ViT-7B/16.  We generalise this to any ViT depth by using
        #    relative depths: [25%, 50%, 75%, 100%] of total layers.
        #
        #    We query backbone.config.num_hidden_layers for the depth and
        #    resolve the four indices at construction time so that they are
        #    logged and stable throughout training.
        # ------------------------------------------------------------------
        total_layers = self.backbone.config.num_hidden_layers
        self.layer_indices = [
            max(1, total_layers // 4),          # ~25% depth
            max(1, total_layers // 2),          # ~50% depth
            max(1, 3 * total_layers // 4),      # ~75% depth
            total_layers,                       # 100% depth (last layer)
        ]
        # dimension:  4 × H  (e.g. 4 × 4096 = 16384 for ViT-7B)
        concat_per_view = NUM_INTERMEDIATE * self.embed_dim  # 16384

        # ------------------------------------------------------------------
        # 3. FEATURE PROJECTION
        #
        #    We offer two strategies for bridging backbone(16k) → encoder(768):
        #
        #    (A) "Original" (concat-then-project): Literal paper implementation.
        #        Concatenate windowed and global maps into a 32,768-dim tensor,
        #        then project.  Peak memory is extremely high (~64GB).
        #
        #    (B) "Memory Efficient" (project-then-sum):
        #        Project each 16k branch to 768 immediately and sum.
        #        Peak memory is ~20x lower (~1.5GB).
        #
        #    BOTH ARE MATHEMATICALLY IDENTICAL for a Linear layer:
        #      Y = [W1, W2] * [X1, X2] + b   ≡   Y = W1*X1 + W2*X2 + b
        # ------------------------------------------------------------------
        if self.memory_efficient:
            self.window_proj = nn.Linear(concat_per_view, ENCODER_DIM)
            self.global_proj = nn.Linear(concat_per_view, ENCODER_DIM)
        else:
            self.feature_proj = nn.Linear(2 * concat_per_view, ENCODER_DIM)

        self.proj_norm = nn.LayerNorm(ENCODER_DIM)

        # ------------------------------------------------------------------
        # 4. ENCODER — stack of 6 self-attention blocks.
        #
        #    Unlike the original DETR that sometimes fuses the encoder
        #    into the backbone, we keep this as a *separate* module so
        #    the backbone stays completely frozen.
        #
        #    Input shape:  (seq_len, batch, ENCODER_DIM)
        #    Output shape: same — a contextualised feature sequence that
        #                  acts as *memory* for the decoder.
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ENCODER_DIM,
            nhead=8,                # 8 attention heads
            dim_feedforward=2048,   # FFN hidden size (standard 4× ratio)
            dropout=0.1,
            batch_first=False,      # expects (seq, batch, dim) convention
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=ENCODER_LAYERS,
        )

        # ------------------------------------------------------------------
        # 5. OBJECT QUERY EMBEDDINGS
        #
        #    We maintain two independent sets of learned query embeddings:
        #      • o2o_query_embed: 1500 queries, each matched to ≤1 GT box
        #        via standard Hungarian bipartite matching at loss time.
        #      • o2m_query_embed: 1500 queries, each matched against k
        #        copies of the GT boxes to improve recall / convergence.
        #
        #    During the decoder's self-attention, we concatenate both
        #    groups and apply an attention mask so the two groups cannot
        #    attend to each other (keeping the matching semantics clean).
        # ------------------------------------------------------------------
        self.o2o_query_embed = nn.Embedding(NUM_QUERIES_O2O, ENCODER_DIM)
        self.o2m_query_embed = nn.Embedding(NUM_QUERIES_O2M, ENCODER_DIM)

        # ------------------------------------------------------------------
        # 6. DECODER — stack of 6 cross-attention blocks.
        #
        #    The combined (NUM_QUERIES_O2O + NUM_QUERIES_O2M) queries
        #    attend to the encoder's output memory.
        # ------------------------------------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=ENCODER_DIM,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=DECODER_LAYERS,
        )

        # ------------------------------------------------------------------
        # 7. PREDICTION HEADS (shared between O2O and O2M queries)
        #
        #    class_embed : Linear → (num_classes + 1) logits
        #                  The "+1" is the "no-object" background class.
        #    bbox_embed  : 3-layer MLP → 4 coordinates [cx, cy, w, h]
        #                  sigmoid-normalised to [0, 1].
        # ------------------------------------------------------------------
        self.class_embed = nn.Linear(ENCODER_DIM, num_classes + 1)

        self.bbox_embed = nn.Sequential(
            nn.Linear(ENCODER_DIM, ENCODER_DIM),
            nn.ReLU(),
            nn.Linear(ENCODER_DIM, ENCODER_DIM),
            nn.ReLU(),
            nn.Linear(ENCODER_DIM, 4),   # [cx, cy, w, h], normalised to [0,1]
        )

    # -----------------------------------------------------------------------
    # HELPER: _extract_intermediate_features
    #
    #   Runs a single image tensor through the frozen backbone and returns
    #   the concatenated features from the 4 selected intermediate layers.
    #
    #   Input:
    #     x       : Tensor of shape (B, 3, H_px, W_px)
    #   Returns:
    #     features: Tensor of shape (B, num_patches, 4 * embed_dim)
    #               where num_patches = (H_px/PATCH_SIZE) × (W_px/PATCH_SIZE)
    # -----------------------------------------------------------------------
    def _extract_intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate exactly how many patch tokens we expect for this image size.
        # This allows us to skip special tokens (CLS, Registers) regardless
        # of the backbone's specific internal configuration.
        B, C, H, W = x.shape
        ph, pw = H // PATCH_SIZE, W // PATCH_SIZE
        num_patches = ph * pw

        # Ask the backbone to return the hidden states of ALL intermediate
        # layers, not just the final one.
        outputs = self.backbone(x, output_hidden_states=True)

        # hidden_states is a tuple of length (num_layers + 1):
        #   index 0      → patch embedding layer output
        #   index 1..N   → transformer block outputs (1-indexed)
        # We select the 4 target layers by absolute index.
        selected = [
            outputs.hidden_states[idx]   # shape: (B, 1 + registers + num_patches, H)
            for idx in self.layer_indices
        ]

        # Take only the LAST num_patches tokens. This skips the leading CLS
        # token and any "register" tokens (common in DINOv2/v3).
        # After this, each tensor has shape: (B, num_patches, H)
        selected = [feat[:, -num_patches:, :] for feat in selected]

        # Concatenate along the channel/embedding dimension.
        # Result shape: (B, num_patches, 4 * H)
        return torch.cat(selected, dim=-1)

    # -----------------------------------------------------------------------
    # HELPER: _build_self_attn_mask
    #
    #   Builds a boolean attention mask for the decoder's self-attention
    #   step that prevents the O2O and O2M query groups from attending
    #   to each other.
    #
    #   The mask has shape (total_queries, total_queries).
    #   A True entry at position [i, j] means query i is NOT ALLOWED to
    #   attend to query j (PyTorch convention: True = masked out).
    # -----------------------------------------------------------------------
    def _build_self_attn_mask(self, device: torch.device) -> torch.Tensor:
        total = NUM_QUERIES_O2O + NUM_QUERIES_O2M
        # Start with all False (every query can attend to every other).
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)

        # Block the O2O group [0:NUM_QUERIES_O2O] from attending to the
        # O2M group [NUM_QUERIES_O2O:total], and vice-versa.
        mask[:NUM_QUERIES_O2O, NUM_QUERIES_O2O:] = True
        mask[NUM_QUERIES_O2O:, :NUM_QUERIES_O2O] = True
        return mask  # shape: (total, total)

    # -----------------------------------------------------------------------
    # FORWARD PASS
    #
    #   Implements the full pipeline:
    #     (A) 3×3 windowing → extract intermediate features → reassemble
    #     (B) global downscaled view → extract → upsample to match (A)
    #     (C) concatenate (A) and (B) channel-wise  → project → encode
    #     (D) decode with dual-query set  → predict classes & boxes
    # -----------------------------------------------------------------------
    def forward(self, images: torch.Tensor):
        """
        Args:
            images : Tensor of shape (B, 3, H_px, W_px).

        Returns:
            dict with keys:
                "pred_logits"     : (B, NUM_QUERIES_O2O, num_classes+1)
                "pred_boxes"      : (B, NUM_QUERIES_O2O, 4)
                "pred_logits_o2m" : (B, NUM_QUERIES_O2M, num_classes+1)
                "pred_boxes_o2m"  : (B, NUM_QUERIES_O2M, 4)
        """
        B, _, H_px, W_px = images.shape
        device = images.device

        # ---------------------------------------------------------------
        # (A) 3×3 WINDOWED BACKBONE PASS
        #
        #   Following the paper (windowing strategy section):
        #   "The image is divided into 3×3 non-overlapping windows.
        #    Each window is forwarded through the backbone..."
        #
        #   Steps:
        #   1. Divide the image into NUM_WINDOWS_GRID × NUM_WINDOWS_GRID
        #      non-overlapping tiles using unfold.
        #   2. Pass each tile through the backbone (reusing weights).
        #   3. Reassemble the tile feature maps into one large spatial map.
        # ---------------------------------------------------------------
        grid = NUM_WINDOWS_GRID   # 3
        tile_H = H_px // grid     # height of each window tile (pixels)
        tile_W = W_px // grid     # width of each window tile  (pixels)

        # Number of patch tokens along each spatial dim for a single tile.
        tile_ph = tile_H // PATCH_SIZE   # patch rows per tile
        tile_pw = tile_W // PATCH_SIZE   # patch cols per tile

        # Collect per-tile feature tensors.
        windowed_tiles = []
        for row in range(grid):
            for col in range(grid):
                # Crop the (row, col) tile from every image in the batch.
                tile = images[:, :, row*tile_H:(row+1)*tile_H, col*tile_W:(col+1)*tile_W]

                # Extract 4-layer intermediate features for this tile.
                # Output: (B, tile_ph * tile_pw, 16384 for ViT-7B)
                tile_feats = self._extract_intermediate_features(tile)

                if self.memory_efficient:
                    # OPTIMIZATION: Project to 768-dim immediately to save memory.
                    tile_feats = self.window_proj(tile_feats)

                windowed_tiles.append(tile_feats)

        # Reassemble the spatial window map.
        rows_of_tiles = []
        for row in range(grid):
            # If memory_efficient, dimensions are (B, tile_ph, tile_pw, 768)
            # Otherwise, dimensions are (B, tile_ph, tile_pw, 16384)
            row_tiles = [
                windowed_tiles[row * grid + col].view(B, tile_ph, tile_pw, -1)
                for col in range(grid)
            ]
            rows_of_tiles.append(torch.cat(row_tiles, dim=2))

        windowed_map = torch.cat(rows_of_tiles, dim=1)
        # windowed_map: (B, grid*tile_ph, grid*tile_pw, C) where C is 768 or 16k.

        full_ph, full_pw = grid * tile_ph, grid * tile_pw

        # ---------------------------------------------------------------
        # (B) GLOBAL DOWNSCALED PASS
        # ---------------------------------------------------------------
        global_resized = F.interpolate(images, size=(tile_H, tile_W), mode="bilinear", align_corners=False)
        global_feats   = self._extract_intermediate_features(global_resized)

        if self.memory_efficient:
            global_map = self.global_proj(global_feats).view(B, tile_ph, tile_pw, -1)
        else:
            global_map = global_feats.view(B, tile_ph, tile_pw, -1)

        # Upsample global_map spatial dimensions to match windowed_map.
        global_map = global_map.permute(0, 3, 1, 2)
        global_map = F.interpolate(global_map, size=(full_ph, full_pw), mode="bilinear", align_corners=False)
        global_map = global_map.permute(0, 2, 3, 1)

        # ---------------------------------------------------------------
        # (C) COMBINING FOR ENCODER
        # ---------------------------------------------------------------
        if self.memory_efficient:
            # Both maps are already 768-dim. Summing is mathematically identical
            # to concatenating and then projecting if the linear layers match.
            projected = self.proj_norm(windowed_map + global_map)
        else:
            # Literal concatenation to doubling features (e.g. 32k dim), then project.
            combined  = torch.cat([windowed_map, global_map], dim=-1)
            projected = self.proj_norm(self.feature_proj(combined))

        # Flatten into token sequence for Transformer blocks.
        seq_len   = full_ph * full_pw
        projected = projected.view(B, seq_len, ENCODER_DIM)

        # Transformer expects (seq_len, B, ENCODER_DIM).
        memory = projected.permute(1, 0, 2)   # (seq_len, B, ENCODER_DIM)

        # Pass through the 6-layer self-attention Transformer encoder.
        # Output shape: still (seq_len, B, ENCODER_DIM)
        if self.training and self.use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            memory = checkpoint(self.transformer_encoder, memory, use_reentrant=False)
        else:
            memory = self.transformer_encoder(memory)

        # ---------------------------------------------------------------
        # (D) DUAL-QUERY DECODER
        #
        #   Concatenate the O2O and O2M query embeddings and run them
        #   jointly through the decoder while masking cross-group
        #   self-attention.
        # ---------------------------------------------------------------
        # Build query sequence: (NUM_QUERIES_O2O + NUM_QUERIES_O2M, B, ENCODER_DIM)
        o2o_q = self.o2o_query_embed.weight.unsqueeze(1).expand(-1, B, -1)  # (1500, B, D)
        o2m_q = self.o2m_query_embed.weight.unsqueeze(1).expand(-1, B, -1)  # (1500, B, D)
        queries = torch.cat([o2o_q, o2m_q], dim=0)                          # (3000, B, D)

        # Build the self-attention mask that blocks cross-group attention.
        # Shape: (3000, 3000) — True entries are masked out.
        self_attn_mask = self._build_self_attn_mask(device)

        # Run the decoder:
        #   tgt    = queries (3000, B, D)
        #   memory = encoder output (seq_len, B, D)
        #   tgt_mask = self-attention mask for the query self-attention
        # Output shape: (3000, B, ENCODER_DIM)
        hs = self.transformer_decoder(
            tgt=queries,
            memory=memory,
            tgt_mask=self_attn_mask,
        )

        # Rearrange to (B, 3000, ENCODER_DIM) for the prediction heads.
        hs = hs.permute(1, 0, 2)   # (B, 3000, ENCODER_DIM)

        # Split back into O2O and O2M portions.
        hs_o2o = hs[:, :NUM_QUERIES_O2O, :]    # (B, 1500, ENCODER_DIM)
        hs_o2m = hs[:, NUM_QUERIES_O2O:, :]    # (B, 1500, ENCODER_DIM)

        # ---------------------------------------------------------------
        # (E) PREDICTION HEADS — applied independently to each group.
        #
        #   class_embed: Linear(ENCODER_DIM → num_classes+1)
        #   bbox_embed:  MLP(ENCODER_DIM → 4), sigmoid → [0,1]
        # ---------------------------------------------------------------
        # One-to-one predictions (used at inference and primary loss).
        outputs_class     = self.class_embed(hs_o2o)                    # (B, 1500, C+1)
        outputs_coord     = self.bbox_embed(hs_o2o).sigmoid()           # (B, 1500, 4)

        # One-to-many predictions (auxiliary training signal only).
        outputs_class_o2m = self.class_embed(hs_o2m)                    # (B, 1500, C+1)
        outputs_coord_o2m = self.bbox_embed(hs_o2m).sigmoid()           # (B, 1500, 4)

        return {
            # Primary outputs — used at inference.
            "pred_logits":     outputs_class,
            "pred_boxes":      outputs_coord,
            # Auxiliary outputs — used only during training for O2M loss.
            "pred_logits_o2m": outputs_class_o2m,
            "pred_boxes_o2m":  outputs_coord_o2m,
        }


# ---------------------------------------------------------------------------
# FACTORY FUNCTION
#
#   Convenience wrapper around the PlainDETRFramework constructor.
# ---------------------------------------------------------------------------
def load_model(
    model_name=CONFIG["train"]["dino_backbone"],
    num_classes=CONFIG["train"]["num_classes"],
    frozen_backbone=True,
    token=HF_TOKEN,
    memory_efficient=None,
):
    """Instantiate and return a PlainDETRFramework model."""
    model = PlainDETRFramework(model_name, num_classes, frozen_backbone, token, memory_efficient)
    return model


# ---------------------------------------------------------------------------
# INFERENCE UTILITY
#
#   Runs a single image through the model, applies a confidence threshold,
#   and optionally visualises the retained detections with matplotlib.
#
#   NOTE: Only the O2O predictions are used at inference time, as per the
#   one-to-one matching convention.
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, image_tensor, threshold=0.5, display=True):
    """
    Args:
        model        : PlainDETRFramework instance (must be on the right device).
        image_tensor : Tensor of shape (3, H, W), a single image (no batch dim).
        threshold    : Confidence threshold for keeping detections.
        display      : If True, render detections on the image with matplotlib.

    Returns:
        boxes  : Tensor of retained box coordinates [cx, cy, w, h] normalised.
        labels : Tensor of predicted class ids for the retained boxes.
    """
    model.eval()

    # Add a batch dimension → (1, 3, H, W)
    outputs = model(image_tensor.unsqueeze(0))

    # Convert O2O class logits to probabilities; drop the background class.
    # probas shape: (num_queries, num_classes)
    probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]

    # Keep only detections whose maximum class probability exceeds threshold.
    keep = probas.max(-1).values > threshold

    # Filter boxes and labels using the keep mask.
    boxes  = outputs["pred_boxes"][0, keep]     # (N_kept, 4)
    labels = probas[keep].argmax(-1)            # (N_kept,)

    print(f"Detections above threshold: {keep.sum().item()}")
    print(boxes)
    print(labels)

    if display:
        plt.figure(figsize=(10, 10))
        img = image_tensor.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        ax = plt.gca()

        for box, label in zip(boxes, labels):
            cx, cy, w, h = box.tolist()

            # Convert from normalised [cx, cy, w, h] to pixel [x, y, bw, bh].
            x  = (cx - w / 2) * img.shape[1]
            y  = (cy - h / 2) * img.shape[0]
            bw = w * img.shape[1]
            bh = h * img.shape[0]

            rect = patches.Rectangle(
                (x, y), bw, bh,
                linewidth=2, edgecolor="r", facecolor="none",
            )
            ax.add_patch(rect)
            plt.text(x, y, f"Class: {label.item()}", color="white", backgroundcolor="red")

        plt.show()

    return boxes, labels