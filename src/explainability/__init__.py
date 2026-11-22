"""
Explainability Module for Cotton Leaf AI Platform.
Provides Gradient-weighted Class Activation Mapping (Grad-CAM)
and publication-ready visualization generation for foliar disease diagnostics.
"""

from src.explainability.gradcam import GradCAM
from src.explainability.visualization import (
    generate_heatmap_image,
    generate_overlay_image,
    generate_three_panel_comparison,
    save_explanation_artifacts,
    SCIENTIFIC_DISCLAIMER,
)

__all__ = [
    "GradCAM",
    "generate_heatmap_image",
    "generate_overlay_image",
    "generate_three_panel_comparison",
    "save_explanation_artifacts",
    "SCIENTIFIC_DISCLAIMER",
]
