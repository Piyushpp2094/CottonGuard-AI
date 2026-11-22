"""
Factory functions for instantiating and loading Cotton Leaf AI models.
"""

from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn

from src.models.classifier import CottonLeafClassifier


def build_classifier(
    model_name: str = "efficientnet_b0",
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: float = 0.3,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> CottonLeafClassifier:
    """
    Constructs a CottonLeafClassifier, optionally loading trained weights from a checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If loading from custom checkpoint, we don't need torchvision pretrained weights
    use_pretrained = pretrained if checkpoint_path is None else False

    model = CottonLeafClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=use_pretrained,
        dropout=dropout
    )

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        # Strip any 'module.' prefix from distributed training if present
        clean_state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(clean_state_dict)

    model.to(device)
    return model
