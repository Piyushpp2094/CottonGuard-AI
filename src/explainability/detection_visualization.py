"""
Detection Visualization Module for CottonGuard AI Faster R-CNN (Phase 9.2B).

Provides non-destructive visual overlay generation for object detection:
- Original cotton leaf photograph
- Ground-truth bounding boxes (standard green)
- Predicted bounding boxes (color-coded by disease class with confidence score)
- Three-panel comparative figures: [Original | Ground Truth | Predictions]
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 6 Cotton disease foreground classes
FOREGROUND_CLASSES = [
    "Bacterial Blight",
    "Curl Virus",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation",
]

# High-contrast color palette for predicted disease classes (RGB)
CLASS_COLORS = {
    0: (220, 38, 38),     # Bacterial Blight: Crimson Red
    1: (168, 85, 247),    # Curl Virus: Purple
    2: (234, 88, 12),     # Herbicide Growth Damage: Orange
    3: (14, 165, 233),    # Leaf Hopper Jassids: Sky Blue
    4: (190, 18, 60),     # Leaf Redding: Deep Rose
    5: (16, 185, 129),    # Leaf Variegation: Emerald
}

GT_COLOR = (34, 197, 94)  # Lime Green for ground-truth boxes


def draw_bounding_boxes(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    scores: Optional[Sequence[float]] = None,
    class_names: Optional[Dict[int, str]] = None,
    box_color: Optional[Tuple[int, int, int]] = None,
    line_width: int = 3,
    score_threshold: float = 0.0,
) -> Image.Image:
    """
    Draws bounding boxes and labels onto a copy of an RGB PIL Image.

    Args:
        image: Source PIL Image in RGB format.
        boxes: List/array of [x1, y1, x2, y2] pixel coordinates.
        labels: List/array of integer class IDs (0-based or 1-based).
        scores: Optional list/array of float confidence scores [0.0, 1.0].
        class_names: Dict mapping class ID to string name.
        box_color: If specified, all boxes use this RGB color. Otherwise uses CLASS_COLORS.
        line_width: Border width in pixels.
        score_threshold: Minimum confidence score to render.

    Returns:
        New RGB PIL Image with rendered bounding boxes.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    if class_names is None:
        class_names = {i: name for i, name in enumerate(FOREGROUND_CLASSES)}

    # Fallback to default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes):
        if len(box) != 4:
            continue
        score = float(scores[i]) if scores is not None and i < len(scores) else None
        if score is not None and score < score_threshold:
            continue

        lbl = int(labels[i]) if i < len(labels) else 0
        # If labels are 1-based (TorchVision convention 1..6), convert to 0..5
        cls_idx = lbl - 1 if lbl in range(1, 7) and lbl not in class_names else lbl
        cls_name = class_names.get(lbl, class_names.get(cls_idx, f"Class {lbl}"))

        color = box_color if box_color is not None else CLASS_COLORS.get(cls_idx, (255, 255, 0))

        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Label text
        label_text = f"{cls_name}: {score:.2f}" if score is not None else cls_name
        # Draw background banner for label text readability
        text_bbox = draw.textbbox((x1, max(0.0, y1 - 14)), label_text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1 + 2, max(0.0, y1 - 14)), label_text, fill=(255, 255, 255), font=font)

    return canvas


def create_detection_comparison_panel(
    image: Image.Image,
    gt_boxes: Sequence[Sequence[float]],
    gt_labels: Sequence[int],
    pred_boxes: Sequence[Sequence[float]],
    pred_labels: Sequence[int],
    pred_scores: Sequence[float],
    title: str = "Faster R-CNN Detection Comparison",
    score_threshold: float = 0.3,
    save_path: Optional[Union[str, Path]] = None,
) -> Image.Image:
    """
    Constructs a side-by-side 3-panel figure:
    [Panel 1: Original Leaf] | [Panel 2: Ground Truth Boxes] | [Panel 3: Predictions]

    Args:
        image: Original RGB PIL image.
        gt_boxes: Ground truth [x1, y1, x2, y2] boxes.
        gt_labels: Ground truth class IDs.
        pred_boxes: Predicted [x1, y1, x2, y2] boxes.
        pred_labels: Predicted class IDs.
        pred_scores: Predicted confidence scores.
        title: Figure title.
        score_threshold: Minimum confidence score to display predictions.
        save_path: Optional path to save the output PNG.

    Returns:
        PIL Image of the stitched 3-panel figure.
    """
    orig_rgb = image.convert("RGB")

    # Panel 2: Ground Truth
    gt_img = draw_bounding_boxes(
        orig_rgb,
        boxes=gt_boxes,
        labels=gt_labels,
        scores=None,
        box_color=GT_COLOR,
        line_width=3,
    )

    # Panel 3: Predictions
    pred_img = draw_bounding_boxes(
        orig_rgb,
        boxes=pred_boxes,
        labels=pred_labels,
        scores=pred_scores,
        box_color=None,
        line_width=3,
        score_threshold=score_threshold,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    axes[0].imshow(orig_rgb)
    axes[0].set_title(f"Original Leaf Image\n(Size: {orig_rgb.size[0]}x{orig_rgb.size[1]})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt_img)
    axes[1].set_title(f"Ground Truth Annotations\n({len(gt_boxes)} boxes)", fontsize=11, fontweight="bold", color="darkgreen")
    axes[1].axis("off")

    valid_preds = sum(1 for s in pred_scores if s >= score_threshold)
    axes[2].imshow(pred_img)
    axes[2].set_title(f"Faster R-CNN Predictions\n({valid_preds} boxes >= {score_threshold:.2f} conf)", fontsize=11, fontweight="bold", color="darkred")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    # Convert matplotlib figure to PIL Image
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba_buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
    panel_pil = Image.fromarray(rgba_buf[:, :, :3], mode="RGB")
    plt.close(fig)

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        panel_pil.save(p, "PNG")

    return panel_pil
