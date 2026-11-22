"""
Prediction History API routes (supports both /api/predictions/history and /api/history).
"""

import csv
import io
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.auth import get_current_user
from backend.app.database import get_db
from backend.app.models import Prediction, User
from backend.app.schemas import PaginatedHistoryResponse, PredictionDetailResponse, PredictionHistoryItem
from backend.app.services.audit_service import record_audit_event
from backend.app.services.prediction_service import serialize_prediction_detail
from src.utils.paths import resolve_platform_path

router = APIRouter(tags=["History"])


def _build_filtered_history_query(
    db: Session,
    current_user: User,
    search: Optional[str] = None,
    disease: Optional[str] = None,
    severity: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    query = db.query(Prediction)
    if current_user.role != "admin":
        query = query.filter(Prediction.user_id == current_user.id)

    if search and search.strip():
        s = f"%{search.strip()}%"
        query = query.filter(
            (Prediction.predicted_class.ilike(s))
            | (Prediction.severity_level.ilike(s))
            | (Prediction.image_filename.ilike(s))
        )

    if disease and disease.strip() and disease.strip().upper() != "ALL":
        query = query.filter(Prediction.predicted_class == disease.strip())

    if severity and severity.strip() and severity.strip().upper() != "ALL":
        query = query.filter(Prediction.severity_level == severity.strip())

    if start_date:
        try:
            dt_start = datetime.fromisoformat(start_date.strip())
            query = query.filter(Prediction.created_at >= dt_start)
        except ValueError:
            pass

    if end_date:
        try:
            dt_end = datetime.fromisoformat(end_date.strip())
            if len(end_date.strip()) == 10:
                dt_end = dt_end.replace(hour=23, minute=59, second=59)
            query = query.filter(Prediction.created_at <= dt_end)
        except ValueError:
            pass

    return query.order_by(Prediction.created_at.desc())


def _to_history_item(p: Prediction) -> Dict[str, Any]:
    img_name = p.image_filename or resolve_platform_path(p.image_path).name
    return {
        "id": p.id,
        "image_url": f"/artifacts/predictions/{img_name}",
        "predicted_class": p.predicted_class,
        "confidence": round(float(p.confidence), 4),
        "affected_area_percentage": p.affected_area_percentage,
        "severity_level": p.severity_level,
        "created_at": p.created_at,
        "detection_count": p.detection.detection_count if p.detection else None,
        "gradcam_comparison_url": p.result.gradcam_comparison_path if p.result else None,
        "is_cotton_leaf": getattr(p, "is_cotton_leaf", True),
        "verification_status": getattr(p, "verification_status", "COTTON_LEAF_VERIFIED"),
        "rejection_reason": getattr(p, "rejection_reason", None),
    }


@router.get("/api/predictions/history", response_model=List[PredictionHistoryItem])
def get_prediction_history(
    search: Optional[str] = Query(None, description="Filter by condition, severity, or filename"),
    disease: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns matching prediction history list sorted chronologically descending."""
    query = _build_filtered_history_query(db, current_user, search, disease, severity, start_date, end_date)
    predictions = query.limit(limit).all()
    return [_to_history_item(p) for p in predictions]


@router.get("/api/history", response_model=PaginatedHistoryResponse)
def get_paginated_history(
    search: Optional[str] = Query(None),
    disease: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(12, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Paginated prediction history with disease, severity, search, and date range filtering."""
    query = _build_filtered_history_query(db, current_user, search, disease, severity, start_date, end_date)
    total = query.count()
    total_pages = max(1, math.ceil(total / page_size)) if total > 0 else 1
    predictions = query.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "items": [_to_history_item(p) for p in predictions],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


def _csv_cell(value: Any) -> str:
    """Formats scalar CSV cell safely against spreadsheet formula injection."""
    if value is None:
        return ""
    s = str(value)
    return f"'{s}" if s[:1] in ("=", "+", "-", "@") else s


@router.get("/api/history/export/csv")
def export_history_csv(
    search: Optional[str] = Query(None),
    disease: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Exports filtered prediction history as a downloadable CSV spreadsheet."""
    rows = _build_filtered_history_query(db, current_user, search, disease, severity, start_date, end_date).limit(1000).all()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "prediction_id",
        "image_filename",
        "predicted_class",
        "confidence_pct",
        "severity_level",
        "affected_area_pct",
        "leaf_pixels",
        "disease_pixels",
        "healthy_pixels",
        "detection_count",
        "model_version",
        "created_at_utc",
    ])
    for p in rows:
        seg = p.segmentation
        det = p.detection
        writer.writerow([
            p.id,
            _csv_cell(p.image_filename),
            _csv_cell(p.predicted_class),
            round(float(p.confidence) * 100.0, 2),
            _csv_cell(p.severity_level),
            p.affected_area_percentage if p.affected_area_percentage is not None else "",
            seg.total_leaf_pixels if seg and seg.total_leaf_pixels is not None else "",
            seg.disease_pixels if seg and seg.disease_pixels is not None else "",
            seg.healthy_pixels if seg and seg.healthy_pixels is not None else "",
            det.detection_count if det else "",
            _csv_cell(p.model_version),
            p.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        ])

    record_audit_event(
        db,
        action="HISTORY_CSV_EXPORT",
        resource="history_export",
        details=f"Exported {len(rows)} scan record(s)",
        user=current_user,
        commit=True,
    )
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="cotton_leaf_history.csv"'},
    )


@router.get("/api/history/{prediction_id}", response_model=PredictionDetailResponse)
def get_history_item_detail(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieves full stored diagnostic breakdown for a specific scan."""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction #{prediction_id} not found.",
        )
    if prediction.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this scan.",
        )
    return serialize_prediction_detail(prediction)


@router.delete("/api/history/{prediction_id}")
def delete_history_item(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Deletes a scan record if owned by current user or administrator."""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction #{prediction_id} not found.",
        )
    if prediction.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this scan.",
        )

    summary = f"Deleted scan #{prediction_id} ({prediction.predicted_class})"
    db.delete(prediction)
    record_audit_event(
        db,
        action="PREDICTION_DELETED",
        resource=f"prediction:{prediction_id}",
        details=summary,
        user=current_user,
    )
    db.commit()
    return {"status": "deleted", "id": prediction_id}
