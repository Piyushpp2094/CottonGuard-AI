"""
Affected Area Estimation and Segmentation Mask Processing for Cotton Leaf AI.

Implements standard pixel-level quantification:
  Affected Area (%) = (Diseased Leaf Pixels / Total Visible Leaf Pixels) * 100

Features:
- Background exclusion (pixel class 0).
- Diseased tissue restriction to the detected leaf boundary.
- Mask dimension validation.
- Visual mask generation: leaf boundary mask, diseased region mask, and colored overlay.
- Minimum leaf size validation to prevent noise amplification.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image


MIN_LEAF_PIXELS: int = 500


def calculate_affected_area_from_mask(
    mask_array: np.ndarray,
    min_leaf_pixels: int = MIN_LEAF_PIXELS,
    leaf_mask_override: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, int, bool, str, None]]:
    """
    Computes affected area metrics from a 2D integer mask array where:
      0 = Background
      1 = Healthy Leaf
      2 = Diseased Tissue

    Enforces:
      - Background exclusion (class 0 ignored in denominator and numerator).
      - Diseased pixels restricted strictly within the visible leaf boundary.
      - Dimension validation when an explicit leaf mask is provided.

    Returns dictionary containing pixel counts, affected percentage, and status.
    """
    if mask_array.ndim != 2:
        raise ValueError(f"Expected 2D mask array, got shape {mask_array.shape}")

    if leaf_mask_override is not None:
        if leaf_mask_override.shape != mask_array.shape:
            raise ValueError(
                f"Leaf mask shape {leaf_mask_override.shape} does not match prediction mask {mask_array.shape}"
            )
        valid_leaf = leaf_mask_override >= 1
    else:
        valid_leaf = mask_array >= 1

    background_pixels = int(np.sum(~valid_leaf))
    disease_pixels = int(np.sum((mask_array == 2) & valid_leaf))
    healthy_pixels = int(np.sum(valid_leaf)) - disease_pixels
    leaf_pixels = healthy_pixels + disease_pixels

    if leaf_pixels < min_leaf_pixels:
        return {
            "status": "INSUFFICIENT_LEAF_AREA",
            "message": f"Detected leaf area ({leaf_pixels} px) is below minimum threshold ({min_leaf_pixels} px).",
            "background_pixels": background_pixels,
            "leaf_pixels": leaf_pixels,
            "healthy_pixels": healthy_pixels,
            "disease_pixels": disease_pixels,
            "affected_area_percentage": None,
            "is_valid": False,
        }

    affected_pct = float((disease_pixels / leaf_pixels) * 100.0)
    affected_pct = round(float(np.clip(affected_pct, 0.0, 100.0)), 2)

    return {
        "status": "CALCULATED",
        "message": "Affected area successfully computed from segmentation mask.",
        "background_pixels": background_pixels,
        "leaf_pixels": leaf_pixels,
        "healthy_pixels": healthy_pixels,
        "disease_pixels": disease_pixels,
        "affected_area_percentage": affected_pct,
        "is_valid": True,
    }


def generate_segmentation_overlays(
    image: Image.Image,
    mask_array: np.ndarray,
    alpha: float = 0.40
) -> Dict[str, Image.Image]:
    """
    Generates PIL visualizations from an RGB image and a 2D segmentation mask:
      1. 'leaf_mask': Grayscale mask of the overall leaf.
      2. 'disease_mask': Grayscale mask of the diseased tissue.
      3. 'overlay': Original image with green tint on healthy leaf and red tint on diseased lesions.
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    
    # Ensure mask matches image dimensions
    if mask_array.shape != (h, w):
        mask_img = Image.fromarray(mask_array.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)
        mask_array = np.array(mask_img)

    # 1. Leaf Mask (255 for leaf pixels, 0 for background)
    leaf_binary = ((mask_array >= 1) * 255).astype(np.uint8)
    leaf_mask_img = Image.fromarray(leaf_binary, mode="L")

    # 2. Disease Mask (255 for diseased pixels, 0 elsewhere)
    disease_binary = ((mask_array == 2) * 255).astype(np.uint8)
    disease_mask_img = Image.fromarray(disease_binary, mode="L")

    # 3. Color Overlay
    img_arr = np.array(img_rgb, dtype=np.float32)
    overlay_arr = img_arr.copy()

    # Healthy leaf tint (Green: [34, 197, 94])
    healthy_idx = (mask_array == 1)
    overlay_arr[healthy_idx] = (
        (1.0 - alpha) * img_arr[healthy_idx] + alpha * np.array([34, 197, 94], dtype=np.float32)
    )

    # Diseased tissue tint (Red: [239, 68, 68])
    disease_idx = (mask_array == 2)
    overlay_arr[disease_idx] = (
        (1.0 - alpha) * img_arr[disease_idx] + alpha * np.array([239, 68, 68], dtype=np.float32)
    )

    overlay_img = Image.fromarray(np.clip(overlay_arr, 0, 255).astype(np.uint8), mode="RGB")

    return {
        "leaf_mask": leaf_mask_img,
        "disease_mask": disease_mask_img,
        "overlay": overlay_img,
    }
