import os
import random
import glob
import argparse
import torch
from torchvision import transforms
from PIL import Image

from src.training.dino_object_detection import PlainDETRFramework, run_inference
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
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint) # In case it's just the state dict
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
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
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_pil = Image.open(random_img_path).convert("RGB")
    img_tensor = transform(img_pil).to(device)

    # 5. Run inference
    print("Running inference and displaying results...")
    run_inference(model, img_tensor, threshold=args.threshold, display=True)

if __name__ == "__main__":
    main()
