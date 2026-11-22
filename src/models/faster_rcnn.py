"""
TorchVision Faster R-CNN ResNet-50 FPN V2 Model Factory for Cotton Leaf AI (Phase 9.2A).

Features:
- State-of-the-art Faster R-CNN V2 architecture with improved FPN and RoIAlign.
- Pretrained weights from COCO_V1 (BSD-3-Clause permissive).
- Replaces standard 91-class RoI box predictor with 7-class FastRCNNPredictor
  (0: background, 1..6: cotton leaf disease classes).
- Tunable trainable_backbone_layers (default: 3 layers: res3, res4, res5).
- Safe device placement with graceful CPU fallback.
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.models.ignore_region_faster_rcnn import equip_faster_rcnn_with_ignore_regions

NUM_DETECTION_CLASSES = 7  # 0: background + 6 foreground disease classes


def build_faster_rcnn_model(
    num_classes: int = NUM_DETECTION_CLASSES,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    enable_ignore_regions: bool = False,
    rpn_ignore_threshold: float = 0.30,
    roi_ignore_threshold: float = 0.30,
) -> FasterRCNN:
    """
    Constructs a Faster R-CNN ResNet-50 FPN V2 model.

    Args:
        num_classes: Total classes including background label 0 (default: 7).
        pretrained: If True, initializes backbone, FPN and RPN with COCO_V1 weights.
        trainable_backbone_layers: Number of trainable ResNet layers from 0 to 5 (default: 3).
        checkpoint_path: Path to custom trained checkpoint to load state_dict from.
        device: Target torch.device (defaults to CUDA if available, else CPU).
        enable_ignore_regions: If True, equips RPN and RoI heads with Tier 2 ignore-region masking.
        rpn_ignore_threshold: IoU/IoA threshold for RPN anchor ignore masking (default: 0.30).
        roi_ignore_threshold: IoU/IoA threshold for RoI proposal ignore masking (default: 0.30).

    Returns:
        Configured FasterRCNN model on the requested device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained and checkpoint_path is None else None

    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # Replace FastRCNNPredictor head for cotton disease classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if enable_ignore_regions:
        model = equip_faster_rcnn_with_ignore_regions(
            model,
            rpn_ignore_threshold=rpn_ignore_threshold,
            roi_ignore_threshold=roi_ignore_threshold,
        )

    if checkpoint_path is not None:
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")
        state = torch.load(ckpt_p, map_location=device)
        state_dict = state.get("model_state_dict", state)
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)

    model.to(device)
    return model
