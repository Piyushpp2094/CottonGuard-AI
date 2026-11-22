"""
Dashboard statistics, temporal trends, and model readiness API routes.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.auth import get_current_user
from backend.app.database import get_db
from backend.app.models import Prediction, ReportRecord, User
from backend.app.routes.history import _to_history_item
from backend.app.schemas import DashboardStatistics
from backend.app.services.prediction_service import prediction_service

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


def _compute_dashboard_payload(current_user: User, db: Session) -> Dict[str, Any]:
    pred_query = db.query(Prediction)
    rep_query = db.query(ReportRecord)
    if current_user.role != "admin":
        pred_query = pred_query.filter(Prediction.user_id == current_user.id)
        rep_query = rep_query.filter(ReportRecord.user_id == current_user.id)

    predictions = pred_query.order_by(Prediction.created_at.desc()).all()
    reports = rep_query.order_by(ReportRecord.generated_at.desc()).limit(5).all()

    total = len(predictions)
    healthy_count = sum(1 for p in predictions if p.predicted_class == "Healthy Leaf" and getattr(p, "is_cotton_leaf", True))
    diseased_count = sum(1 for p in predictions if p.predicted_class != "Healthy Leaf" and getattr(p, "is_cotton_leaf", True) is True)

    condition_counts = Counter(p.predicted_class for p in predictions)
    severity_counts = Counter(p.severity_level for p in predictions)

    # Group chronological scan counts by UTC date
    daily: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "healthy": 0, "diseased": 0})
    for p in predictions:
        day_key = p.created_at.strftime("%Y-%m-%d")
        daily[day_key]["total"] += 1
        if p.predicted_class == "Healthy Leaf":
            daily[day_key]["healthy"] += 1
        else:
            daily[day_key]["diseased"] += 1

    trends: List[Dict[str, Any]] = [
        {"date": d, "total": vals["total"], "healthy": vals["healthy"], "diseased": vals["diseased"]}
        for d, vals in sorted(daily.items())[-14:]
    ]

    recent_items = [_to_history_item(p) for p in predictions[:6]]
    recent_reports = [
        {
            "id": r.id,
            "prediction_id": r.prediction_id,
            "predicted_class": r.prediction.predicted_class if r.prediction else "Unknown",
            "file_size_bytes": r.file_size_bytes,
            "generated_at": r.generated_at,
        }
        for r in reports
    ]

    return {
        "total_analyzed": total,
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "condition_distribution": dict(condition_counts),
        "severity_distribution": dict(severity_counts),
        "recent_predictions": recent_items,
        "trends": trends,
        "model_readiness": prediction_service.get_model_readiness(),
        "recent_reports": recent_reports,
    }


@router.get("/statistics", response_model=DashboardStatistics)
def get_dashboard_statistics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns summary KPIs, trends, model readiness, and recent scans."""
    return _compute_dashboard_payload(current_user, db)


@router.get("/stats", response_model=DashboardStatistics)
def get_dashboard_stats_alias(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Alias route for /api/dashboard/statistics matching unified REST convention."""
    return _compute_dashboard_payload(current_user, db)
