"""
Visualization utilities for Grad-CAM Explainability in Cotton Leaf AI.

Provides modular methods for:
- Converting 2D normalized activation maps into RGB heatmaps
- Generating blended overlays with configurable alpha blending
- Constructing publication-quality three-panel visual comparisons
  ([Original Image | Grad-CAM Heatmap | Diagnostic Overlay]) with metadata and scientific disclaimers
- Persisting structured explainability artifacts (images and JSON metadata)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPLAINABILITY_DIR = PROJECT_ROOT / "artifacts" / "explainability"

SCIENTIFIC_DISCLAIMER = (
    "Grad-CAM highlights image regions that contributed to the model's prediction. "
    "It does not represent a pixel-accurate disease segmentation or disease severity measurement."
)


def get_colormap(cmap_name: str = "jet"):
    """Safely retrieves a matplotlib colormap across library versions."""
    try:
        return matplotlib.colormaps[cmap_name]
    except (AttributeError, KeyError):
        return plt.get_cmap(cmap_name)


def generate_heatmap_image(
    heatmap: np.ndarray,
    colormap: str = "jet"
) -> Image.Image:
    """
    Transforms a 2D float32 activation map in [0.0, 1.0] into an RGB PIL Image.

    Args:
        heatmap: 2D NumPy array of shape (H, W) with values normalized to [0.0, 1.0].
        colormap: Matplotlib colormap name (default 'jet', also 'turbo', 'viridis').

    Returns:
        PIL Image in RGB mode.
    """
    # Ensure values are within [0.0, 1.0]
    cam_clipped = np.clip(heatmap, 0.0, 1.0).astype(np.float32)

    cmap = get_colormap(colormap)
    rgba = cmap(cam_clipped)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)

    return Image.fromarray(rgb, mode="RGB")


def generate_overlay_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet"
) -> Image.Image:
    """
    Blends an RGB leaf photograph with a Grad-CAM activation heatmap.

    Overlay formula:
        Overlay = (1 - alpha) * Original + alpha * Heatmap

    Args:
        original_image: Source leaf photograph (PIL Image).
        heatmap: 2D NumPy array of shape (H, W) normalized to [0.0, 1.0].
        alpha: Overlay opacity weight in [0.0, 1.0] (default 0.4).
        colormap: Matplotlib colormap name.

    Returns:
        Blended PIL Image in RGB mode matching original image dimensions.
    """
    rgb_orig = original_image.convert("RGB")
    orig_w, orig_h = rgb_orig.size

    # Ensure heatmap matches original image dimensions
    heatmap_img = generate_heatmap_image(heatmap, colormap=colormap)
    if heatmap_img.size != (orig_w, orig_h):
        heatmap_img = heatmap_img.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)

    orig_arr = np.array(rgb_orig, dtype=np.float32)
    heat_arr = np.array(heatmap_img, dtype=np.float32)

    alpha_clamped = max(0.0, min(1.0, float(alpha)))
    blended_arr = (1.0 - alpha_clamped) * orig_arr + alpha_clamped * heat_arr
    blended_uint8 = np.clip(blended_arr, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(blended_uint8, mode="RGB")


def generate_three_panel_comparison(
    original_image: Image.Image,
    heatmap_img: Image.Image,
    overlay_img: Image.Image,
    predicted_class: str,
    confidence: float,
    target_class: Optional[str] = None,
    image_name: Optional[str] = None
) -> Image.Image:
    """
    Constructs a side-by-side 3-panel visualization:
    [ Original Photograph | Grad-CAM Heatmap | Diagnostic Overlay ]
    with academic presentation banner, confidence score, and scientific disclaimer.

    Args:
        original_image: Original leaf photograph.
        heatmap_img: Colorized Grad-CAM heatmap.
        overlay_img: Blended overlay image.
        predicted_class: Predicted condition name.
        confidence: Prediction confidence probability (0.0 to 1.0).
        target_class: Target condition name if diagnostic mode used.
        image_name: Optional filename label.

    Returns:
        Composite PIL Image with top header banner and bottom disclaimer.
    """
    rgb_orig = original_image.convert("RGB")
    rgb_heat = heatmap_img.convert("RGB")
    rgb_over = overlay_img.convert("RGB")

    # Standardize panel display height for uniform alignment
    panel_height = 360
    aspect_ratio = rgb_orig.width / max(1, rgb_orig.height)
    panel_width = max(240, int(panel_height * aspect_ratio))

    panel1 = rgb_orig.resize((panel_width, panel_height), resample=Image.Resampling.BILINEAR)
    panel2 = rgb_heat.resize((panel_width, panel_height), resample=Image.Resampling.BILINEAR)
    panel3 = rgb_over.resize((panel_width, panel_height), resample=Image.Resampling.BILINEAR)

    # Layout dimensions
    margin = 16
    spacing = 16
    header_height = 80
    panel_title_height = 32
    footer_height = 48

    total_width = margin * 2 + panel_width * 3 + spacing * 2
    total_height = header_height + panel_title_height + panel_height + footer_height + margin * 2

    # Canvas
    canvas = Image.new("RGB", (total_width, total_height), color=(248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # --- Header Banner ---
    draw.rectangle(
        [(0, 0), (total_width, header_height)],
        fill=(15, 23, 42)  # Dark slate
    )
    title_text = "COTTON LEAF AI — GRAD-CAM SALIENCY EXPLAINABILITY"
    draw.text((margin, 14), title_text, fill=(241, 245, 249), font=font)

    info_str = f"Predicted: {predicted_class} ({confidence * 100:.2f}%)"
    if target_class and target_class != predicted_class:
        info_str += f" | Explaining Target: {target_class}"
    if image_name:
        info_str += f" | Image: {image_name}"
    draw.text((margin, 38), info_str, fill=(74, 222, 128), font=font)  # Green accent

    model_subtext = "Architecture: EfficientNet-B0 | Target Layer: features[-1] (Conv2dNormActivation 1280 channels)"
    draw.text((margin, 58), model_subtext, fill=(148, 163, 184), font=font)

    # --- Panel Titles ---
    y_titles = header_height + margin
    titles = [
        "1. Original Leaf Photograph",
        "2. Grad-CAM Attention Heatmap",
        "3. Diagnostic Foliar Overlay"
    ]
    x_positions = [
        margin,
        margin + panel_width + spacing,
        margin + (panel_width + spacing) * 2
    ]

    for x_pos, title in zip(x_positions, titles):
        draw.text((x_pos, y_titles), title, fill=(30, 41, 59), font=font)

    # --- Paste Panels ---
    y_panels = y_titles + panel_title_height
    canvas.paste(panel1, (x_positions[0], y_panels))
    canvas.paste(panel2, (x_positions[1], y_panels))
    canvas.paste(panel3, (x_positions[2], y_panels))

    # Borders around panels
    for x_pos in x_positions:
        draw.rectangle(
            [(x_pos - 1, y_panels - 1), (x_pos + panel_width, y_panels + panel_height)],
            outline=(203, 213, 225),
            width=1
        )

    # --- Footer Banner with Scientific Disclaimer ---
    y_footer = y_panels + panel_height + margin
    draw.rectangle(
        [(margin, y_footer), (total_width - margin, y_footer + footer_height)],
        fill=(254, 243, 199),  # Amber tint
        outline=(245, 158, 11),
        width=1
    )
    disclaimer_label = "[SCIENTIFIC NOTICE]"
    draw.text((margin + 12, y_footer + 8), disclaimer_label, fill=(180, 83, 9), font=font)
    draw.text((margin + 12, y_footer + 24), SCIENTIFIC_DISCLAIMER, fill=(146, 64, 14), font=font)

    return canvas


def save_explanation_artifacts(
    original_image: Image.Image,
    heatmap: np.ndarray,
    metadata: Dict[str, Any],
    base_name: str,
    output_dir: Optional[Path] = None,
    alpha: float = 0.4,
    colormap: str = "jet"
) -> Dict[str, str]:
    """
    Saves the full suite of explainability artifacts:
    - Original photograph -> artifacts/explainability/original/<base_name>.jpg
    - Heatmap -> artifacts/explainability/heatmaps/<base_name>_heatmap.png
    - Overlay -> artifacts/explainability/overlays/<base_name>_overlay.png
    - Comparison -> artifacts/explainability/comparisons/<base_name>_comparison.png
    - Metadata -> artifacts/explainability/reports/<base_name>_metadata.json

    Returns:
        Dictionary of absolute filesystem paths and relative URL paths for all artifacts.
    """
    base_dir = output_dir if output_dir is not None else EXPLAINABILITY_DIR
    orig_dir = base_dir / "original"
    heat_dir = base_dir / "heatmaps"
    over_dir = base_dir / "overlays"
    comp_dir = base_dir / "comparisons"
    repo_dir = base_dir / "reports"

    for d in [orig_dir, heat_dir, over_dir, comp_dir, repo_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Images
    rgb_orig = original_image.convert("RGB")
    heatmap_img = generate_heatmap_image(heatmap, colormap=colormap)
    overlay_img = generate_overlay_image(rgb_orig, heatmap, alpha=alpha, colormap=colormap)
    comparison_img = generate_three_panel_comparison(
        original_image=rgb_orig,
        heatmap_img=heatmap_img,
        overlay_img=overlay_img,
        predicted_class=metadata.get("predicted_class_name", "Unknown"),
        confidence=metadata.get("confidence", 0.0),
        target_class=metadata.get("target_class_name"),
        image_name=base_name
    )

    orig_path = orig_dir / f"{base_name}.jpg"
    heat_path = heat_dir / f"{base_name}_heatmap.png"
    over_path = over_dir / f"{base_name}_overlay.png"
    comp_path = comp_dir / f"{base_name}_comparison.png"
    meta_path = repo_dir / f"{base_name}_metadata.json"

    rgb_orig.save(orig_path, quality=95)
    heatmap_img.save(heat_path)
    overlay_img.save(over_path)
    comparison_img.save(comp_path)

    # 2. Enrich and save metadata JSON
    full_metadata = {
        "image_filename": f"{base_name}.jpg",
        "predicted_class": metadata.get("predicted_class_name"),
        "prediction_confidence": metadata.get("confidence"),
        "target_class_index": metadata.get("target_class_idx"),
        "target_class_name": metadata.get("target_class_name"),
        "model_architecture": "EfficientNet-B0",
        "checkpoint_path": str(PROJECT_ROOT / "artifacts" / "checkpoints" / "classification" / "best_model.pt"),
        "target_convolutional_layer": "backbone.features[-1] (Conv2dNormActivation 320->1280)",
        "input_dimensions": metadata.get("input_resolution", [224, 224]),
        "heatmap_dimensions": metadata.get("heatmap_resolution", list(rgb_orig.size)),
        "overlay_alpha": alpha,
        "colormap": colormap,
        "device": metadata.get("device", "unknown"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
        "all_probabilities": metadata.get("all_probabilities", {}),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=2)

    return {
        "original_path": str(orig_path),
        "heatmap_path": str(heat_path),
        "overlay_path": str(over_path),
        "comparison_path": str(comp_path),
        "metadata_path": str(meta_path),
        "original_url": f"/artifacts/explainability/original/{base_name}.jpg",
        "heatmap_url": f"/artifacts/explainability/heatmaps/{base_name}_heatmap.png",
        "overlay_url": f"/artifacts/explainability/overlays/{base_name}_overlay.png",
        "comparison_url": f"/artifacts/explainability/comparisons/{base_name}_comparison.png",
        "metadata_url": f"/artifacts/explainability/reports/{base_name}_metadata.json",
    }
