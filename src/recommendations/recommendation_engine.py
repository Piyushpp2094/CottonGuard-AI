"""
Farmer Recommendation Engine for Cotton Leaf AI.

Provides verified agronomic decision-support guidance based on ICAR-CICR advisories,
integrating model predictions, confidence levels, and estimated severity.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECOMMENDATIONS_PATH = PROJECT_ROOT / "configs" / "recommendations.json"


class RecommendationEngine:
    """
    Offline agricultural guidance engine supporting all seven cotton leaf conditions.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_RECOMMENDATIONS_PATH
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Recommendations config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.classes_data: Dict[str, Any] = self.data.get("classes", {})
        self.disclaimer: str = self.data.get("disclaimer", "")

    def get_recommendation(
        self,
        predicted_class: str,
        confidence: float,
        severity_level: str = "UNAVAILABLE",
        affected_area_percentage: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generates tailored decision support recommendations for a prediction.
        """
        if predicted_class not in self.classes_data:
            return {
                "status": "UNKNOWN_CONDITION",
                "condition": predicted_class,
                "confidence": round(float(confidence), 4),
                "severity_level": severity_level,
                "first_steps": ["Consult your local agricultural extension service for diagnosis."],
                "preventive_practices": ["Maintain standard good agricultural practices."],
                "avoid": "Do not apply chemical sprays without professional diagnosis.",
                "disclaimer": self.disclaimer
            }

        base_info = self.classes_data[predicted_class]

        first_steps = list(base_info.get("first_steps", []))
        preventive_practices = list(base_info.get("preventive_practices", []))
        symptoms = list(base_info.get("symptoms", []))
        avoid_guidance = base_info.get("avoid", "")
        pathogen_or_cause = base_info.get("pathogen_or_cause", "")

        # Incorporate severity-specific priority actions
        priority_action = None
        if severity_level == "HIGH":
            priority_action = (
                "High foliar damage detected (>40% visible leaf area affected): Arrange an immediate in-field "
                "scouting inspection with your local KVK or extension officer to assess whole-field threshold impact."
            )
            first_steps.insert(0, priority_action)
        elif severity_level == "MODERATE":
            priority_action = (
                "Moderate foliar symptoms detected (15% - 40% leaf area affected): Monitor adjacent plants "
                "and record whether symptom severity increases over the next 3 to 5 days."
            )
            first_steps.insert(0, priority_action)

        # Incorporate confidence warnings
        confidence_warning = None
        if confidence < 0.60:
            confidence_warning = (
                f"Model prediction confidence ({confidence*100:.1f}%) is below recommended threshold (60%). "
                "Capture clearer, well-lit photos of multiple leaves and seek in-person expert confirmation."
            )

        return {
            "status": "SUCCESS",
            "condition": predicted_class,
            "pathogen_or_cause": pathogen_or_cause,
            "confidence": round(float(confidence), 4),
            "severity_level": severity_level,
            "affected_area_percentage": affected_area_percentage,
            "confidence_warning": confidence_warning,
            "priority_action": priority_action,
            "symptoms_to_observe": symptoms,
            "immediate_first_steps": first_steps,
            "preventive_management": preventive_practices,
            "critical_warning": avoid_guidance,
            "disclaimer": self.disclaimer,
        }
