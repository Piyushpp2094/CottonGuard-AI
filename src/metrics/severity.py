"""
Configurable Severity Estimation Module for CottonGuard AI.

Strictly separates three independent concepts:
  1. Classification confidence (EfficientNet-B0 softmax certainty)
  2. Percentage of affected leaf area (U-Net pixel ratio)
  3. Agricultural severity category (provisional 6-tier scale pending field calibration)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "severity_rules.yaml"


class SeverityEstimator:
    """Evaluates foliar damage severity using configurable agricultural thresholds."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if self.config_path.is_file():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = {}

        self.category_validated = bool(self.config.get("category_validated", False))
        self.thresholds_verified = bool(self.config.get("thresholds_verified", self.category_validated))
        self.calculation_method = str(self.config.get("calculation_method", "unet_pixel_ratio_v1"))
        self.threshold_basis = self.config.get(
            "threshold_basis",
            "Provisional digital pixel-ratio baselines pending manual cotton field calibration.",
        )

        # 6-tier provisional bounds
        six_tier = self.config.get("provisional_6_tier_thresholds", {})
        self.none_max = float(six_tier.get("none_max_percentage", 0.0))
        self.trace_max = float(six_tier.get("trace_max_percentage", 10.0))
        self.mild_max = float(six_tier.get("mild_max_percentage", 25.0))
        self.mod6_max = float(six_tier.get("moderate_max_percentage", 50.0))
        self.severe_max = float(six_tier.get("severe_max_percentage", 75.0))

        # Legacy 4-tier bounds preserved for backward compatibility
        legacy = self.config.get("thresholds", {})
        self.no_lesion_max = float(legacy.get("no_lesion_max_percentage", 0.5))
        self.low_max = float(legacy.get("low_max_percentage", 15.0))
        self.moderate_max = float(legacy.get("moderate_max_percentage", 40.0))
        self.min_confidence = float(self.config.get("confidence_gates", {}).get("min_classification_confidence", 0.60))

    def _classify_6_tier(self, pct: float) -> str:
        if pct <= self.none_max:
            return "None"
        if pct <= self.trace_max:
            return "Trace"
        if pct <= self.mild_max:
            return "Mild"
        if pct <= self.mod6_max:
            return "Moderate"
        if pct <= self.severe_max:
            return "Severe"
        return "Very Severe"

    def evaluate(
        self,
        affected_area_percentage: Optional[float],
        predicted_class: str,
        confidence: float,
        leaf_pixels: Optional[int] = None,
        disease_pixels: Optional[int] = None,
        healthy_pixels: Optional[int] = None,
        segmentation_model_version: Optional[str] = None,
        no_leaf_detected: bool = False,
    ) -> Dict[str, Any]:
        """
        Determines both the literature-aligned 6-tier `severity_category` (`category_validated=false`)
        and legacy `level` without conflating confidence with lesion percentage.
        """
        if no_leaf_detected:
            return {
                "level": "UNAVAILABLE",
                "severity_category": "UNAVAILABLE",
                "category_validated": self.category_validated,
                "calculation_status": "no_leaf_detected",
                "calculation_method": self.calculation_method,
                "model_version": segmentation_model_version or "UNAVAILABLE",
                "label": "No cotton leaf detected in segmentation mask",
                "affected_area_percentage": None,
                "affected_leaf_area_percentage": None,
                "total_leaf_pixels": leaf_pixels or 0,
                "leaf_pixels": leaf_pixels or 0,
                "diseased_pixels": 0,
                "disease_pixels": 0,
                "healthy_pixels": 0,
                "classification_confidence": round(float(confidence), 4),
                "confidence": round(float(confidence), 4),
                "predicted_class": predicted_class,
                "segmentation_model_version": segmentation_model_version or "UNAVAILABLE",
                "thresholds_verified": self.thresholds_verified,
                "is_available": False,
                "note": "No valid cotton leaf region detected (below minimum leaf pixel threshold).",
            }

        if affected_area_percentage is None:
            return {
                "level": "UNAVAILABLE",
                "severity_category": "UNAVAILABLE",
                "category_validated": self.category_validated,
                "calculation_status": "model_unavailable",
                "calculation_method": self.calculation_method,
                "model_version": segmentation_model_version or "UNAVAILABLE",
                "label": "Severity calculation requires a trained segmentation model.",
                "affected_area_percentage": None,
                "affected_leaf_area_percentage": None,
                "total_leaf_pixels": leaf_pixels,
                "leaf_pixels": leaf_pixels,
                "diseased_pixels": disease_pixels,
                "disease_pixels": disease_pixels,
                "healthy_pixels": healthy_pixels,
                "classification_confidence": round(float(confidence), 4),
                "confidence": round(float(confidence), 4),
                "predicted_class": predicted_class,
                "segmentation_model_version": segmentation_model_version or "UNAVAILABLE",
                "thresholds_verified": self.thresholds_verified,
                "is_available": False,
                "note": "Severity calculation requires a trained segmentation model.",
            }

        uncertain_warning = None
        if confidence < self.min_confidence:
            uncertain_warning = (
                f"Classification confidence ({confidence*100:.1f}%) is below reliable threshold "
                f"({self.min_confidence*100:.1f}%). Field scouting is strongly recommended."
            )

        pct = round(max(0.0, min(100.0, float(affected_area_percentage))), 2)
        severity_category = self._classify_6_tier(pct)

        # Legacy level preserved for backward compatibility
        if pct <= self.no_lesion_max:
            legacy_level = "NONE"
            legacy_label = "No Visible Lesion Detected"
        elif pct <= self.low_max:
            legacy_level = "LOW"
            legacy_label = "Low Severity"
        elif pct <= self.moderate_max:
            legacy_level = "MODERATE"
            legacy_label = "Moderate Severity"
        else:
            legacy_level = "HIGH"
            legacy_label = "High Severity"

        description = (
            f"Affected leaf lamina is {pct:.2f}% -> provisional category '{severity_category}' "
            f"(category_validated={self.category_validated})."
        )

        agronomic_context = []
        if predicted_class == "Leaf Hopper Jassids":
            agronomic_context.append(
                "Foliar symptom area indicates hopper burn extent, but economic threshold level (ETL) "
                "requires counting nymphs/adults on the leaf underside (ETL: 2 nymphs/leaf)."
            )
        elif predicted_class == "Leaf Redding":
            agronomic_context.append(
                "Reddening symptoms often reflect physiological or abiotic stress (nutrient imbalance, "
                "temperature drop, or waterlogging) rather than infectious pathogen spread."
            )
        elif predicted_class == "Healthy Leaf" and pct > self.no_lesion_max:
            agronomic_context.append(
                "Warning: Classification model predicted Healthy Leaf, but segmentation detected surface discoloration. "
                "Inspect leaf in field for early stress or dust deposition."
            )

        return {
            "level": legacy_level,
            "severity_category": severity_category,
            "category_validated": self.category_validated,
            "calculation_status": "success",
            "calculation_method": self.calculation_method,
            "model_version": segmentation_model_version or "U-Net-3Class",
            "label": f"{severity_category} ({legacy_label})" if not self.category_validated else severity_category,
            "affected_area_percentage": pct,
            "affected_leaf_area_percentage": pct,
            "total_leaf_pixels": leaf_pixels,
            "leaf_pixels": leaf_pixels,
            "diseased_pixels": disease_pixels,
            "disease_pixels": disease_pixels,
            "healthy_pixels": healthy_pixels,
            "classification_confidence": round(float(confidence), 4),
            "confidence": round(float(confidence), 4),
            "predicted_class": predicted_class,
            "segmentation_model_version": segmentation_model_version or "U-Net-3Class",
            "thresholds_verified": self.thresholds_verified,
            "threshold_basis": self.threshold_basis,
            "is_available": True,
            "description": description,
            "uncertainty_warning": uncertain_warning,
            "agronomic_notes": agronomic_context,
            "disclaimer": "2D visible image pixel ratio only; distinct from classification confidence and physical leaf area in cm².",
        }
