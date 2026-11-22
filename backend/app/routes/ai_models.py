"""
Modular Component AI API routes:
  - POST /api/classification/predict
  - POST /api/detection/predict
  - POST /api/segmentation/predict
  - POST /api/severity/calculate
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.app.auth import get_current_user
from backend.app.models import User
from backend.app.schemas import (
    ClassificationPredictResponse,
    DetectionPredictResponse,
    SegmentationPredictResponse,
    SeverityCalculateRequest,
)
from backend.app.services.prediction_service import prediction_service

router = APIRouter(prefix="/api", tags=["AI Models & Severity"])


@router.post("/classification/predict", response_model=ClassificationPredictResponse)
def classify_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Runs EfficientNet-B0 7-class foliar condition classification."""
    return prediction_service.classify_uploaded_image(file)


@router.post("/detection/predict", response_model=DetectionPredictResponse)
def detect_disease_regions(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    current_user: User = Depends(get_current_user),
):
    """
    Runs YOLO disease-region localization when trained checkpoint exists.
    Returns an explicit UNAVAILABLE payload if annotations/weights are absent.
    """
    return prediction_service.detect_uploaded_image(
        file=file,
        conf_threshold=max(0.05, min(0.95, conf_threshold)),
        iou_threshold=max(0.10, min(0.90, iou_threshold)),
    )


@router.post("/segmentation/predict", response_model=SegmentationPredictResponse)
def segment_leaf_and_lesions(
    file: UploadFile = File(...),
    alpha: float = Form(0.40),
    current_user: User = Depends(get_current_user),
):
    """
    Runs 3-class U-Net leaf & diseased-tissue segmentation when trained checkpoint exists.
    Returns an explicit UNAVAILABLE payload if annotations/weights are absent.
    """
    return prediction_service.segment_uploaded_image(
        file=file,
        alpha=max(0.0, min(1.0, alpha)),
    )


@router.post("/severity/calculate")
def calculate_severity(
    payload: SeverityCalculateRequest,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Evaluates disease severity from validated segmentation pixel ratios or area percentage.
    """
    area_pct = payload.affected_area_percentage
    if area_pct is None and payload.leaf_pixels and payload.leaf_pixels > 0 and payload.disease_pixels is not None:
        area_pct = round(min(100.0, max(0.0, (payload.disease_pixels / payload.leaf_pixels) * 100.0)), 2)

    return prediction_service.severity_estimator.evaluate(
        affected_area_percentage=area_pct,
        predicted_class=payload.predicted_class,
        confidence=payload.confidence,
        leaf_pixels=payload.leaf_pixels,
        disease_pixels=payload.disease_pixels,
        healthy_pixels=payload.healthy_pixels,
        segmentation_model_version=payload.segmentation_model_version,
    )
