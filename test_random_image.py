import os
import random
import glob
import argparse
import torch
from torchvision import transforms
from PIL import Image

from src.training.dino_object_detection import PlainDETRFramework, run_inference
from src.training.train import load_state_dict_flexible
from src.config.utils import CONFIG

def main():
    parser = argparse.ArgumentParser(description="Test model inference on a random test image")
    parser.add_argument("--image_dir", type=str, default="data/bdd/images/100k/val", help="Path to test/val images directory")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/latest_checkpoint.pth", help="Path to model weights")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for predictions")
    args = parser.parse_args()

    if not os.path.exists(args.image_dir):
        # Fallback to test if val doesn't exist
        fallback_dir = "data/bdd/images/100k/test"
        if os.path.exists(fallback_dir):
            args.image_dir = fallback_dir
        else:
            print(f"Error: Image directory {args.image_dir} not found.")
            return

    # 1. Select a random image
    image_files = glob.glob(os.path.join(args.image_dir, "*.jpg"))
    if not image_files:
        print(f"Error: No images found in {args.image_dir}")
        return
        
    random_img_path = random.choice(image_files)
    print(f"Selected image: {random_img_path}")

    # 2. Load the model
    print("Loading model architecture...")
    model = PlainDETRFramework(
        model_name=CONFIG["train"]["dino_backbone"],
        num_classes=CONFIG["train"]["num_classes"],
        frozen_backbone=True
    )

    # 3. Load checkpoint
    if os.path.exists(args.checkpoint_path):
        print(f"Loading weights from {args.checkpoint_path}...")
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            
            # --- Robust Prefix Handling & Loading ---
            # 1. First check for architectural mismatch (memory_efficient flag)
            has_ckpt_window = any("window_proj" in k for k in state_dict)
            has_ckpt_feature = any("feature_proj" in k for k in state_dict)
            model_has_window = hasattr(model, "window_proj")
            
            if has_ckpt_feature and model_has_window:
                print("  → Warning: Checkpoint likely used feature_proj (memory_efficient=False) but model was built with window_proj (memory_efficient=True).")
                print("  → Reloading model with memory_efficient=False to match checkpoint...")
                model = PlainDETRFramework(
                    model_name=CONFIG["train"]["dino_backbone"],
                    num_classes=CONFIG["train"]["num_classes"],
                    frozen_backbone=True,
                    memory_efficient=False
                )
            elif has_ckpt_window and not model_has_window:
                print("  → Warning: Checkpoint likely used window_proj (memory_efficient=True) but model was built with feature_proj (memory_efficient=False).")
                print("  → Reloading model with memory_efficient=True to match checkpoint...")
                model = PlainDETRFramework(
                    model_name=CONFIG["train"]["dino_backbone"],
                    num_classes=CONFIG["train"]["num_classes"],
                    frozen_backbone=True,
                    memory_efficient=True
                )

            # 2. Use flexible loader for prefix handling and final loading
            missing_keys, unexpected_keys = load_state_dict_flexible(model, state_dict)
            
            # Note: We don't use strict=True because of architectural flags like memory_efficient
            # that we've already tried to handle, but reporting mismatches is helpful.
            
            if len(missing_keys) > 0:
                print(f"  → Warning: Missing {len(missing_keys)} keys in checkpoint.")
                print(f"    Missing sample: {list(missing_keys)[:10]}")
            
            if len(unexpected_keys) > 0:
                print(f"  → Warning: Unexpected {len(unexpected_keys)} keys in checkpoint.")
                print(f"    Unexpected sample: {list(unexpected_keys)[:10]}")
            
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint_path}.")
        print("Using untrained weights for testing. Please specify a valid --checkpoint_path if you have one.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # 4. Prepare image
    print("Preparing image for inference...")
    img_pil = Image.open(random_img_path).convert("RGB")
    orig_w, orig_h = img_pil.size
    
    # Maintain aspect ratio: resize so short side is 1024 (or original if smaller)
    # Then round to the nearest multiple of 48 (3 window grid * 16 patch size)
    # to ensure clean tiling.
    target_short_side = 1024
    if orig_h < orig_w:
        new_h = target_short_side
        new_w = int(orig_w * target_short_side / orig_h)
    else:
        new_w = target_short_side
        new_h = int(orig_h * target_short_side / orig_w)
        
    stride = 48
    new_h = (new_h // stride) * stride
    new_w = (new_w // stride) * stride
    
    print(f"Resizing from {orig_w}x{orig_h} to {new_w}x{new_h} (maintaining aspect ratio).")

    transform = transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_pil).to(device)

    # 5. Run inference
    print("Running inference and displaying results...")
    run_inference(model, img_tensor, threshold=args.threshold, display=True)

if __name__ == "__main__":
    main()
