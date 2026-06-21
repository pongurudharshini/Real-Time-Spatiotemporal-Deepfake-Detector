import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from config import Config

class SpatiotemporalViT(nn.Module):
    def __init__(self):
        super().__init__()
        print("Initializing Pre-Trained ViT (Downloading v2 weights if first run)...")
        
        # Upgraded to a much more robust, modern Deepfake Vision Transformer
        self.vit = ViTForImageClassification.from_pretrained(
            "prithivMLmods/Deep-Fake-Detector-v2-Model",
            attn_implementation="eager"
        )
        
        # ImageNet Normalization constants required by pre-trained Vision Transformers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.last_attn_map = None

    def forward(self, x):
        # Input tensor 'x' is 5D: (Batch, Time, Channels, Height, Width)
        B, T, C, H, W = x.shape
        
        # Flatten Time into Batch so the 2D pre-trained model can process all 16 frames simultaneously
        x = x.reshape(B * T, C, H, W)
        
        # Normalize the tensor colors perfectly for the pre-trained weights
        x = (x - self.mean) / self.std
        
        # Run inference and extract attention maps for the XAI heatmaps
        outputs = self.vit(pixel_values=x, output_attentions=True)
        logits = outputs.logits  # Shape: (B*T, 2)
        
        # Spatiotemporal Voting: Average the predictions across all 16 frames to get a consensus
        logits = logits.view(B, T, -1).mean(dim=1)  # Shape: (B, 2)
        
        # Save the attention maps for the Streamlit dashboard overlay
        attentions = outputs.attentions[-1] 
        
        # Re-shape back to temporal dimension and average across time
        self.last_attn_map = attentions.view(B, T, attentions.shape[1], attentions.shape[2], attentions.shape[3]).mean(dim=1)
        
        return logits

    def get_last_attention_map(self):
        # Average across the attention heads and isolate the [CLS] token mapping for the dashboard
        cls_attn = self.last_attn_map.mean(dim=1)[:, 0, 1:] 
        return cls_attn