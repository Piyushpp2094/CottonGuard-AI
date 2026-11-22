"""
Administrator Management & System Monitoring API routes.
Strictly protected by role-based access control (role == 'admin').
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.auth import require_admin
from backend.app.database import get_db
from backend.app.models import AuditLog, Prediction, ReportRecord, User
from backend.app.routes.history import _to_history_item
from backend.app.schemas import (
    AdminOverviewResponse,
    AdminUserSummary,
    AuditLogItem,
    UserResponse,
    UserRoleUpdate,
    UserStatusUpdate,
)
from backend.app.services.audit_service import record_audit_event
from backend.app.services.model_registry import read_model_registry
from backend.app.services.prediction_service import prediction_service

router = APIRouter(prefix="/api/admin", tags=["Admin"])
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@router.get("/overview", response_model=AdminOverviewResponse)
def get_admin_overview(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Returns global platform usage, disease distribution, and model readiness."""
    total_users = db.query(User).count()
    predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
    total_reports = db.query(ReportRecord).count()

    healthy_count = sum(1 for p in predictions if p.predicted_class == "Healthy Leaf" and getattr(p, "is_cotton_leaf", True))
    diseased_count = sum(1 for p in predictions if p.predicted_class != "Healthy Leaf" and getattr(p, "is_cotton_leaf", True) is True)

    return {
        "total_users": total_users,
        "total_predictions": len(predictions),
        "total_reports": total_reports,
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "condition_distribution": dict(Counter(p.predicted_class for p in predictions)),
        "severity_distribution": dict(Counter(p.severity_level for p in predictions)),
        "system_status": prediction_service.get_model_readiness(),
        "recent_predictions": [_to_history_item(p) for p in predictions[:10]],
    }


@router.get("/users", response_model=List[AdminUserSummary])
def list_users(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Lists registered users and their scan counts."""
    counts = dict(
        db.query(Prediction.user_id, func.count(Prediction.id))
        .group_by(Prediction.user_id)
        .all()
    )
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": u.role,
            "is_active": bool(getattr(u, "is_active", True)),
            "prediction_count": int(counts.get(u.id, 0)),
            "created_at": u.created_at,
        }
        for u in users
    ]


@router.patch("/users/{user_id}/role", response_model=UserResponse)
def update_user_role(
    user_id: int,
    payload: UserRoleUpdate,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Updates a user's access role ('farmer' or 'admin')."""
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User #{user_id} not found.")
    if target.id == admin_user.id and payload.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot revoke your own administrator role while logged in.",
        )
    old_role = target.role
    target.role = payload.role
    record_audit_event(
        db,
        action="USER_ROLE_CHANGE",
        resource=f"user:{target.id}",
        details=f"Changed {target.username} role from {old_role} to {payload.role}",
        user=admin_user,
    )
    db.commit()
    db.refresh(target)
    return target


@router.patch("/users/{user_id}/status", response_model=UserResponse)
def update_user_status(
    user_id: int,
    payload: UserStatusUpdate,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Activates or deactivates a user account."""
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User #{user_id} not found.")
    if target.id == admin_user.id and not payload.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own administrator account while signed in.",
        )
    target.is_active = payload.is_active
    record_audit_event(
        db,
        action="USER_STATUS_CHANGE",
        resource=f"user:{target.id}",
        details=f"Set {target.username} is_active={payload.is_active}",
        user=admin_user,
    )
    db.commit()
    db.refresh(target)
    return target


@router.get("/audit-logs", response_model=List[AuditLogItem])
def get_audit_logs(
    action: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Returns chronological platform security and activity audit logs."""
    query = db.query(AuditLog)
    if action and action.strip() and action.strip().upper() != "ALL":
        query = query.filter(AuditLog.action == action.strip().upper())
    return query.order_by(AuditLog.created_at.desc()).limit(limit).all()


def _load_json_safe(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@router.get("/system")
def get_system_diagnostics(
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    """Returns verified dataset, evaluation metrics, annotation, leakage/provenance, and GPU telemetry."""
    manifest_summary = _load_json_safe(PROJECT_ROOT / "artifacts" / "manifests" / "manifest_summary.json")
    cls_full = _load_json_safe(PROJECT_ROOT / "artifacts" / "metrics" / "classification_metrics.json")
    ann_summary = _load_json_safe(PROJECT_ROOT / "artifacts" / "manifests" / "annotation_summary.json")
    leakage_report = _load_json_safe(PROJECT_ROOT / "artifacts" / "metrics" / "dataset_leakage_report.json")
    provenance_report = _load_json_safe(PROJECT_ROOT / "artifacts" / "metrics" / "checkpoint_provenance_report.json")

    metric_charts: Dict[str, str] = {}
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    for key, filename in (
        ("confusion_matrix", "confusion_matrix.png"),
        ("per_class_f1", "per_class_f1.png"),
        ("precision_recall", "precision_recall.png"),
        ("confidence_distribution", "confidence_distribution.png"),
    ):
        if (metrics_dir / filename).is_file():
            metric_charts[key] = f"/artifacts/metrics/{filename}"

    return {
        "readiness": prediction_service.get_model_readiness(),
        "dataset_summary": manifest_summary,
        "classifier_benchmark": cls_full.get("overall_performance", {}),
        "classifier_full_metrics": cls_full,
        "annotation_summary": ann_summary,
        "leakage_report": leakage_report,
        "provenance_report": provenance_report,
        "metric_charts": metric_charts,
    }


@router.get("/models")
def get_admin_models(
    admin_user: User = Depends(require_admin),
) -> Dict[str, Any]:
    """Returns synchronized model registry and live runtime readiness."""
    return {
        "registry": read_model_registry(),
        "runtime": prediction_service.get_model_readiness(),
    }
