from typing import Any
import torch.nn as nn
import torch


class ViTModel(nn.Module):
    def __init__(self, num_classes=2, feature_dim=768, pretrained=False):
        super().__init__()
        try:
            from torchvision.models import vit_b_16
        except Exception:
            raise RuntimeError("torchvision vit_b_16 is required for ViTModel")
        vit_backbone = vit_b_16(weights=None)
        self.conv_proj = vit_backbone.conv_proj
        self.class_token = vit_backbone.class_token
        self.pos_embedding = vit_backbone.encoder.pos_embedding
        self.encoder_blocks = vit_backbone.encoder.layers
        self.encoder_norm = vit_backbone.encoder.ln
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: Any):
        n, c, h, w = x.shape
        x = self.conv_proj(x)
        x = x.reshape(n, self.pos_embedding.shape[-1], -1)
        x = x.permute(0, 2, 1)
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.encoder_blocks(x)
        cls_token_output = x[:, 0]
        cls_token_output = self.encoder_norm(cls_token_output)
        logits = self.classifier(cls_token_output)
        return logits
