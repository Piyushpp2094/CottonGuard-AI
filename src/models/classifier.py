"""
Classification model architectures for Cotton Leaf AI.

Implements transfer learning backbones (primarily EfficientNet-B0)
with modular head customization, backbone freezing/unfreezing,
and Grad-CAM hooks for model explainability.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models


class CottonLeafClassifier(nn.Module):
    """
    Cotton Leaf Disease Classifier based on EfficientNet.
    
    Supports:
    - EfficientNet-B0 (default, optimal for 6GB RTX 3050 VRAM)
    - EfficientNet-B2 (comparison model)
    - Configurable dropout and classification head
    - Backbone layer freezing/unfreezing for staged transfer learning
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.dropout_rate = dropout

        if self.model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
        elif self.model_name == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b2(weights=weights)
            in_features = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}. Use 'efficientnet_b0' or 'efficientnet_b2'.")

        # Replace classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits (B, num_classes)."""
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns normalized softmax probabilities (B, num_classes)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def freeze_backbone(self) -> None:
        """Freezes all convolutional feature extraction layers."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreezes all layers for full end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_last_conv_layer(self) -> nn.Module:
        """Returns the final convolutional feature block (for Grad-CAM explainability)."""
        return self.backbone.features[-1]
