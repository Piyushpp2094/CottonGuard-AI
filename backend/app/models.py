"""
SQLAlchemy ORM models for Cotton Leaf AI.

Defines normalized entities and ownership relationships for:
  - Users
  - Predictions
  - PredictionResult (legacy & Explainability / Grad-CAM paths)
  - DetectionResult (YOLO bounding boxes & metadata)
  - SegmentationResult (U-Net pixel masks & pixel counts)
  - SeverityResult (affected area %, calibration status & category)
  - ReportRecord (generated ReportLab PDF reports)
"""

from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from backend.app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="farmer", nullable=False)  # "farmer" or "admin"
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    reports = relationship("ReportRecord", back_populates="user", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    image_filename = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    predicted_class = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    affected_area_percentage = Column(Float, nullable=True)
    severity_level = Column(String(50), default="UNAVAILABLE", nullable=False)
    model_version = Column(String(50), default="EfficientNet-B0-v1.0", nullable=False)
    status = Column(String(50), default="SUCCESS", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_cotton_leaf = Column(Boolean, nullable=True, default=True)
    verification_status = Column(String(50), default="COTTON_LEAF_VERIFIED", nullable=True)
    verification_confidence = Column(Float, nullable=True)
    rejection_reason = Column(Text, nullable=True)

    user = relationship("User", back_populates="predictions")
    result = relationship("PredictionResult", back_populates="prediction", uselist=False, cascade="all, delete-orphan")
    detection = relationship("DetectionResult", back_populates="prediction", uselist=False, cascade="all, delete-orphan")
    segmentation = relationship("SegmentationResult", back_populates="prediction", uselist=False, cascade="all, delete-orphan")
    severity = relationship("SeverityResult", back_populates="prediction", uselist=False, cascade="all, delete-orphan")
    reports = relationship("ReportRecord", back_populates="prediction", cascade="all, delete-orphan")


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), unique=True, nullable=False)
    leaf_mask_path = Column(String(500), nullable=True)
    disease_mask_path = Column(String(500), nullable=True)
    overlay_path = Column(String(500), nullable=True)
    gradcam_heatmap_path = Column(String(500), nullable=True)
    gradcam_overlay_path = Column(String(500), nullable=True)
    gradcam_comparison_path = Column(String(500), nullable=True)
    gradcam_metadata_path = Column(String(500), nullable=True)
    all_probabilities_json = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    recommendations_json = Column(Text, nullable=False)
    uncertainty_notes = Column(Text, nullable=True)

    prediction = relationship("Prediction", back_populates="result")


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), unique=True, nullable=False)
    detections_json = Column(Text, nullable=False, default="[]")
    detection_count = Column(Integer, default=0, nullable=False)
    annotated_image_path = Column(String(500), nullable=True)
    model_version = Column(String(80), default="UNAVAILABLE", nullable=False)
    status = Column(String(50), default="UNAVAILABLE", nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    prediction = relationship("Prediction", back_populates="detection")


class SegmentationResult(Base):
    __tablename__ = "segmentation_results"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), unique=True, nullable=False)
    leaf_mask_path = Column(String(500), nullable=True)
    disease_mask_path = Column(String(500), nullable=True)
    overlay_path = Column(String(500), nullable=True)
    total_leaf_pixels = Column(Integer, nullable=True)
    disease_pixels = Column(Integer, nullable=True)
    healthy_pixels = Column(Integer, nullable=True)
    affected_area_percentage = Column(Float, nullable=True)
    model_version = Column(String(80), default="UNAVAILABLE", nullable=False)
    status = Column(String(50), default="UNAVAILABLE", nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    prediction = relationship("Prediction", back_populates="segmentation")


class SeverityResult(Base):
    __tablename__ = "severity_results"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), unique=True, nullable=False)
    severity_level = Column(String(50), default="UNAVAILABLE", nullable=False)
    severity_label = Column(String(200), nullable=True)
    affected_area_percentage = Column(Float, nullable=True)
    is_available = Column(Boolean, default=False, nullable=False)
    thresholds_verified = Column(Boolean, default=False, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    prediction = relationship("Prediction", back_populates="severity")


class ReportRecord(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    pdf_filename = Column(String(255), nullable=False)
    pdf_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    prediction = relationship("Prediction", back_populates="reports")
    user = relationship("User", back_populates="reports")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=True)
    username = Column(String(50), default="system", nullable=False)
    action = Column(String(60), index=True, nullable=False)
    resource = Column(String(120), nullable=False)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    jti = Column(String(64), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False, nullable=False)
    replaced_by_jti = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class RevokedToken(Base):
    __tablename__ = "revoked_tokens"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(64), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class PredictionJob(Base):
    __tablename__ = "prediction_jobs"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    status = Column(String(30), default="queued", nullable=False)  # queued|running|completed|failed
    current_stage = Column(String(60), default="queued", nullable=False)
    step = Column(Integer, default=0, nullable=False)
    total_steps = Column(Integer, default=7, nullable=False)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)
    error_code = Column(String(60), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
