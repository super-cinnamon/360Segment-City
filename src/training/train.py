import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

class CustomDetectionDataset(Dataset):
    def __init__(self, img_paths, annotations, transform=None, percentage=1.0):
        """
        Args:
            img_paths: List of paths to images.
            annotations: List of dicts with 'boxes' (N, 4) and 'labels' (N,).
            transform: Standard PyTorch transformations (must resize to ~1024x1024).
        """
        sample_size = int(len(img_paths) * percentage)
        self.img_paths = img_paths[:sample_size]
        self.annotations = annotations[:sample_size]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        target = self.annotations[idx]
        
        if self.transform:
            img = self.transform(img)
            
        # Output requirement:
        # img: Tensor [3, H, W]
        # target: {'boxes': Tensor [N, 4], 'labels': Tensor [N]}
        # Boxes MUST be in [cx, cy, w, h] format, normalized by image size [0, 1].

        return img, target

def train_plain_detr(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    device, 
    epochs=12,               # Standard for Faster RCNN/DETR in the paper
    save_path="./checkpoints", 
    start_epoch=0,
    checkpoint_interval=1    # Save every N epochs
):
    model.to(device)
    
    # Optional: Load latest checkpoint if it exists to resume
    checkpoint_file = os.path.join(save_path, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        
        # Progress bar setup
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, targets in pbar:
            # images: [Batch, 3, H, W]
            # targets: list of dicts with 'boxes' and 'labels'
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Loss Calculation (Hungarian matching loss)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # Backward pass
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")

        # --- CHECKPOINT SAVING ---
        if (epoch + 1) % checkpoint_interval == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            
            # Save periodic and "latest" versions
            torch.save(state, os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pth"))
            torch.save(state, checkpoint_file)
            print(f"Checkpoint saved to {save_path}")

    print("Training finished!")


# --- SAVE SETUP ---
def save_checkpoint(model, path, epoch):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, f"plain_detr_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
