"""
Image preprocessing and data augmentation pipelines for Cotton Leaf AI.

Standardizes image preprocessing for training, validation, testing,
and single-image production inference. Also provides denormalization utilities
to convert normalized PyTorch tensors back to viewable RGB images.
"""

from typing import Tuple, Union
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


# Standard ImageNet normalization parameters used by pretrained torchvision backbones
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE: int = 224


def get_train_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    Returns image transformation pipeline for training with data augmentation.
    
    Includes scale jittering, random crop, flips, slight rotation,
    and subtle color jittering to enhance model generalization without
    distorting diagnostic disease patterns.
    """
    scale_size = int(image_size * 1.143)  # e.g., 256 for 224
    return transforms.Compose([
        transforms.Resize((scale_size, scale_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.08,
            hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    Returns deterministic image transformation pipeline for validation and testing.
    Resizes image directly to target dimensions and applies ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    Returns deterministic transformation pipeline for production inference on uploaded leaf images.
    """
    return get_eval_transforms(image_size=image_size)


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> torch.Tensor:
    """
    Reverses ImageNet normalization on a tensor (C, H, W) or batch (B, C, H, W).
    Clamps output values to [0.0, 1.0].
    """
    device = tensor.device
    mean_tensor = torch.tensor(mean, device=device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)

    denorm = tensor * std_tensor + mean_tensor
    return torch.clamp(denorm, 0.0, 1.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Converts a single (C, H, W) normalized or denormalized PyTorch tensor to a PIL RGB Image.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # If values are outside [0, 1], assume normalized and denormalize
    if tensor.min() < 0.0 or tensor.max() > 1.0:
        tensor = denormalize_tensor(tensor)
    
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")
