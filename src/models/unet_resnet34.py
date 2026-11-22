"""
U-Net with ResNet-34 Encoder & Albumentations Pipeline (Phase 4 — MIT Licensed).

Architecture & Spec:
- Segmentation classes: 0 = Background, 1 = Healthy leaf tissue, 2 = Diseased leaf tissue
- Input resolution: 512 x 512 (RTX 3050 6 GB batch size = 8)
- Loss: L_total = 0.5 * CrossEntropyLoss(weight=[0.3, 1.0, 2.5]) + 0.5 * DiceLoss
- Inference strictly gated on verified checkpoint existence (`NOT_TRAINED` until trained on CVAT masks).
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image

from src.metrics.area_calculation import calculate_affected_area_from_mask
from src.training.segmentation_losses import SoftDiceLoss, compute_segmentation_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNET_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "segmentation" / "best_unet.pt"
SEGMENTATION_CLASSES: Dict[int, str] = {
    0: "Background",
    1: "Healthy leaf tissue",
    2: "Diseased leaf tissue",
}


def get_segmentation_train_transforms(image_size: int = 512) -> A.Compose:
    """Albumentations training pipeline matching CottonGuard AI specification."""
    return A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
            A.GaussNoise(p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.ElasticTransform(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2(),
        ]
    )


def get_segmentation_eval_transforms(image_size: int = 512) -> A.Compose:
    """Deterministic Albumentations evaluation/inference pipeline."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2(),
        ]
    )


class WeightedCombinedDiceCELoss(nn.Module):
    """L_total = 0.5 * CrossEntropyLoss(weight=[0.3, 1.0, 2.5]) + 0.5 * SoftDiceLoss."""

    def __init__(
        self,
        class_weights: Sequence[float] = (0.3, 1.0, 2.5),
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.register_buffer("ce_class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.ce_class_weights.to(device=logits.device, dtype=logits.dtype)
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=weight)
        dice_loss = self.dice(logits, targets)
        return (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)


def build_unet_resnet34(
    num_classes: int = 3,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Instantiates SMP U-Net (ResNet-34 encoder) and optionally loads trained checkpoint weights."""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
    if checkpoint_path is not None:
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"UNet-ResNet34 checkpoint not found: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location=device or "cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)

    if device is not None:
        model = model.to(device)
    return model


class UNetResNet34Segmentor:
    """Gated inference wrapper for 3-class UNet-ResNet34 leaf and disease segmentation."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        image_size: int = 512,
        device: Optional[torch.device] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_UNET_CHECKPOINT
        self.image_size = int(image_size)
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._eval_transform = get_segmentation_eval_transforms(self.image_size)
        self._model: Optional[nn.Module] = None

    @property
    def is_trained(self) -> bool:
        return self.checkpoint_path.is_file()

    @property
    def status(self) -> str:
        return "trained" if self.is_trained else "NOT_TRAINED"

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "UNet-ResNet34",
            "architecture": "unet_resnet34",
            "library": f"segmentation-models-pytorch {smp.__version__} (MIT)",
            "input_size": [self.image_size, self.image_size],
            "classes": dict(SEGMENTATION_CLASSES),
            "status": self.status,
            "checkpoint_path": str(self.checkpoint_path) if self.is_trained else None,
            "annotation_requirements": "CVAT 3-class segmentation masks required before training.",
        }

    def _ensure_loaded(self) -> nn.Module:
        if not self.is_trained:
            raise FileNotFoundError(f"UNet-ResNet34 checkpoint not trained yet at: {self.checkpoint_path}")
        if self._model is None:
            self._model = build_unet_resnet34(
                num_classes=3,
                encoder_name="resnet34",
                encoder_weights=None,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
            self._model.eval()
        return self._model

    def predict(self, rgb_image: Image.Image) -> Dict[str, Any]:
        """Runs 512x512 segmentation if trained, otherwise returns honest NOT_TRAINED payload."""
        if not self.is_trained:
            return {
                "status": "NOT_TRAINED",
                "available": False,
                "model_name": "UNet-ResNet34",
                "model_version": None,
                "affected_leaf_area_percentage": None,
                "total_leaf_pixels": None,
                "diseased_pixels": None,
                "healthy_pixels": None,
                "mask": None,
                "reason": (
                    "UNet-ResNet34 checkpoint is not trained yet. Genuine CVAT pixel masks "
                    "are required before segmentation and severity percentage become active."
                ),
            }

        model = self._ensure_loaded()
        rgb_np = np.array(rgb_image.convert("RGB"), dtype=np.uint8)
        transformed = self._eval_transform(image=rgb_np)
        tensor_cpu = transformed["image"].unsqueeze(0)

        used_device = str(self.device)
        warning = None
        try:
            tensor = tensor_cpu.to(self.device)
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                logits = model(tensor)
                mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpu = torch.device("cpu")
            model.to(cpu)
            try:
                with torch.inference_mode():
                    logits = model(tensor_cpu.to(cpu))
                    mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            finally:
                if self.device.type == "cuda":
                    model.to(self.device)
            used_device = "cpu"
            warning = "CUDA out of memory during UNet-ResNet34 inference; retried safely on CPU."

        orig_w, orig_h = rgb_image.size
        if mask.shape != (orig_h, orig_w):
            mask = np.array(
                Image.fromarray(mask).resize((orig_w, orig_h), Image.Resampling.NEAREST),
                dtype=np.uint8,
            )

        area = calculate_affected_area_from_mask(mask)
        return {
            "status": "success" if area["is_valid"] else area["status"],
            "available": True,
            "model_name": "UNet-ResNet34",
            "model_version": "v1.0.0",
            "inference_device": used_device,
            "warning": warning,
            "affected_leaf_area_percentage": area["affected_area_percentage"],
            "total_leaf_pixels": area["leaf_pixels"],
            "diseased_pixels": area["disease_pixels"],
            "healthy_pixels": area["healthy_pixels"],
            "mask": mask,
            "message": area["message"],
        }


__all__ = [
    "DEFAULT_UNET_CHECKPOINT",
    "SEGMENTATION_CLASSES",
    "UNetResNet34Segmentor",
    "WeightedCombinedDiceCELoss",
    "build_unet_resnet34",
    "compute_segmentation_metrics",
    "get_segmentation_eval_transforms",
    "get_segmentation_train_transforms",
]
