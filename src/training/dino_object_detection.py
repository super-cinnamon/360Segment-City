import os 
import dotenv

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.config.utils import CONFIG


dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

class PlainDETRFramework(nn.Module):
    def __init__(self, model_name=CONFIG["train"]["dino_backbone"], num_classes=CONFIG["train"]["num_classes"], frozen_backbone=True, token=HF_TOKEN):
        super().__init__()
        # 1. Load DINOv3 Backbone (e.g., ViT-L/16 or ViT-7B) 
        self.backbone = AutoModel.from_pretrained(model_name, token=token)
        self.embed_dim = self.backbone.config.hidden_size
        
        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 2. Plain-DETR Head Components 
        # In Plain-DETR, we typically use the patch features from the last layer [cite: 103]
        self.query_embed = nn.Embedding(300, self.embed_dim) # 300 object queries
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # 3. Prediction Heads
        self.class_embed = nn.Linear(self.embed_dim, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 4) # [x, y, w, h]
        )

    def forward(self, images):
        # Extract patch features from DINOv3 [cite: 103]
        outputs = self.backbone(images)
        features = outputs.last_hidden_state[:, 1:, :] # Remove CLS token
        
        # Flatten for Transformer (Sequence Length, Batch, Embed Dim)
        features = features.permute(1, 0, 2)
        
        # Object Queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, images.shape[0], 1)
        
        # Transformer Decoder
        hs = self.transformer_decoder(tgt=query_embed, memory=features)
        hs = hs.permute(1, 0, 2) # (Batch, Queries, Embed Dim)
        
        # Predictions
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return {"pred_logits": outputs_class, "pred_boxes": outputs_coord}


def load_model(model_name=CONFIG["train"]["dino_backbone"], num_classes=CONFIG["train"]["num_classes"], frozen_backbone=True, token=HF_TOKEN):
    model = PlainDETRFramework(model_name, num_classes, frozen_backbone, token)
    return model


@torch.no_grad()
def run_inference(model, image_tensor, threshold=0.5, display=True):
    model.eval()
    outputs = model(image_tensor.unsqueeze(0))
    
    # Process logits to probabilities
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    
    # Filter boxes and labels
    boxes = outputs['pred_boxes'][0, keep]
    labels = probas[keep].argmax(-1)
    
    print(boxes)
    print(labels)

    if display:
        plt.figure(figsize=(10, 10))
        img = image_tensor.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        ax = plt.gca()
        
        for box, label in zip(boxes, labels):
            cx, cy, w, h = box.tolist()
            # Un-normalize to pixel coords
            x, y = (cx - w/2) * img.shape[1], (cy - h/2) * img.shape[0]
            bw, bh = w * img.shape[1], h * img.shape[0]
            
            rect = patches.Rectangle((x, y), bw, bh, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x, y, f"Class: {label.item()}", color='white', backgroundcolor='red')
        plt.show()
        
    return boxes, labels