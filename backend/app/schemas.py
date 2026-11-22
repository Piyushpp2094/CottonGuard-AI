"""
Pydantic schemas for request validation and API serialization in Cotton Leaf AI.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


# User Schemas
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=6, max_length=100)
    role: Optional[str] = "farmer"


class UserLogin(BaseModel):
    username_or_email: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool = True
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    token_type: str = "bearer"
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    refresh_token: Optional[str] = None


class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None


class PredictionJobCreateResponse(BaseModel):
    job_id: str
    status: str
    stream_url: str
    total_steps: int = 7


class PredictionJobStatusResponse(BaseModel):
    job_id: str
    status: str
    current_stage: str
    step: int
    total_steps: int
    prediction_id: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class UserRoleUpdate(BaseModel):
    role: str = Field(..., pattern="^(farmer|admin)$")


class UserStatusUpdate(BaseModel):
    is_active: bool


# Detection & Segmentation Component Schemas
class DetectionBox(BaseModel):
    bbox: List[float]
    normalized_bbox: List[float] = Field(default_factory=list)
    class_id: int
    class_name: str
    confidence: float
    box: Optional[List[float]] = None
    label: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def fill_compat_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Map box <-> bbox
            if "bbox" not in data and "box" in data:
                data["bbox"] = data["box"]
            elif "box" not in data and "bbox" in data:
                data["box"] = data["bbox"]
            # Map label <-> class_id
            if "class_id" not in data and "label" in data:
                data["class_id"] = data["label"]
            elif "label" not in data and "class_id" in data:
                data["label"] = data["class_id"]
            # Ensure normalized_bbox is a list
            if "normalized_bbox" not in data or data["normalized_bbox"] is None:
                data["normalized_bbox"] = []
        return data


class DetectionPredictResponse(BaseModel):
    status: str
    available: bool
    model_version: str
    detection_count: int = 0
    detections: List[DetectionBox] = []
    annotated_image_url: Optional[str] = None
    message: Optional[str] = None


class SegmentationPredictResponse(BaseModel):
    status: str
    available: bool
    model_version: str
    affected_area_percentage: Optional[float] = None
    leaf_pixels: Optional[int] = None
    disease_pixels: Optional[int] = None
    healthy_pixels: Optional[int] = None
    overlay_url: Optional[str] = None
    leaf_mask_url: Optional[str] = None
    disease_mask_url: Optional[str] = None
    message: Optional[str] = None


class ClassificationPredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    model_version: str
    device: str
    latency_ms: float
    warning: Optional[str] = None


class SeverityCalculateRequest(BaseModel):
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    affected_area_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    leaf_pixels: Optional[int] = Field(None, ge=0)
    disease_pixels: Optional[int] = Field(None, ge=0)
    healthy_pixels: Optional[int] = Field(None, ge=0)
    segmentation_model_version: Optional[str] = None


# Prediction History & Unified Analysis Schemas
class PredictionHistoryItem(BaseModel):
    id: int
    image_url: str
    predicted_class: str
    confidence: float
    affected_area_percentage: Optional[float] = None
    severity_level: str
    created_at: datetime
    detection_count: Optional[int] = None
    gradcam_comparison_url: Optional[str] = None
    is_cotton_leaf: Optional[bool] = True
    verification_status: Optional[str] = "COTTON_LEAF_VERIFIED"
    rejection_reason: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PaginatedHistoryResponse(BaseModel):
    items: List[PredictionHistoryItem]
    total: int
    page: int
    page_size: int
    total_pages: int


class PredictionDetailResponse(BaseModel):
    id: int
    user_id: int
    image_url: str
    predicted_class: str
    confidence: float
    all_probabilities: Optional[Dict[str, float]] = None
    affected_area_percentage: Optional[float] = None
    severity_level: str
    severity_label: Optional[str] = None
    thresholds_verified: bool = False
    model_version: str
    status: str
    created_at: datetime
    latency_ms: Optional[float] = None

    # Detection outputs
    detection_status: str = "UNAVAILABLE"
    detection_model_version: str = "UNAVAILABLE"
    detection_count: int = 0
    detections: List[DetectionBox] = []
    detection_image_url: Optional[str] = None
    detection_notes: Optional[str] = None

    # Segmentation outputs
    segmentation_status: str = "UNAVAILABLE"
    segmentation_model_version: str = "UNAVAILABLE"
    leaf_pixels: Optional[int] = None
    disease_pixels: Optional[int] = None
    healthy_pixels: Optional[int] = None
    overlay_url: Optional[str] = None
    leaf_mask_url: Optional[str] = None
    disease_mask_url: Optional[str] = None

    # Grad-CAM outputs
    gradcam_heatmap_url: Optional[str] = None
    gradcam_overlay_url: Optional[str] = None
    gradcam_comparison_url: Optional[str] = None

    # Recommendations & Notes
    recommendations: Dict[str, Any]
    uncertainty_notes: Optional[str] = None
    inference_device: Optional[str] = None
    warning: Optional[str] = None

    # Cotton Leaf Verification Gate & Image Quality fields
    is_cotton_leaf: Optional[bool] = True
    verification_status: Optional[str] = "COTTON_LEAF_VERIFIED"
    verification_confidence: Optional[float] = None
    rejection_reason: Optional[str] = None
    user_message: Optional[str] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    probabilities: Optional[Dict[str, float]] = None

    model_config = ConfigDict(from_attributes=True)


# Explainability Schema
class GradCAMResponse(BaseModel):
    predicted_class: str
    confidence: float
    target_class: str
    original_url: str
    heatmap_url: str
    overlay_url: str
    comparison_url: str
    metadata_url: str
    target_layer: str
    model_architecture: str
    device: str
    scientific_disclaimer: str
    all_probabilities: Dict[str, float]


# Dashboard & Admin Schemas
class TrendPoint(BaseModel):
    date: str
    total: int
    healthy: int
    diseased: int


class ReportSummaryItem(BaseModel):
    id: int
    prediction_id: int
    predicted_class: str
    file_size_bytes: int
    generated_at: datetime


class DashboardStatistics(BaseModel):
    total_analyzed: int
    healthy_count: int
    diseased_count: int
    condition_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]
    recent_predictions: List[PredictionHistoryItem]
    trends: List[TrendPoint] = []
    model_readiness: Dict[str, Any] = {}
    recent_reports: List[ReportSummaryItem] = []


class AdminUserSummary(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool = True
    prediction_count: int
    created_at: datetime


class AuditLogItem(BaseModel):
    id: int
    user_id: Optional[int] = None
    username: str
    action: str
    resource: str
    details: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AdminOverviewResponse(BaseModel):
    total_users: int
    total_predictions: int
    total_reports: int
    healthy_count: int
    diseased_count: int
    condition_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]
    system_status: Dict[str, Any]
    recent_predictions: List[PredictionHistoryItem]
