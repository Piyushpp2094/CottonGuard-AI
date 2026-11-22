"""
PDF Report Generation & Download API route.
"""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.auth import get_current_user
from backend.app.config import get_settings
from backend.app.database import get_db
from backend.app.models import Prediction, ReportRecord, User
from backend.app.services.audit_service import record_audit_event
from backend.app.services.report_service import generate_pdf_report

router = APIRouter(prefix="/api/reports", tags=["Reports"])


def _ensure_report_file(prediction_id: int, current_user: User, db: Session) -> ReportRecord:
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction #{prediction_id} not found.",
        )

    # Cotton Leaf Verification Gate: strictly prevent agronomic reports for invalid/rejected inputs
    is_cotton = getattr(prediction, "is_cotton_leaf", None)
    pred_status = getattr(prediction, "status", "")
    if is_cotton is False or pred_status.startswith("REJECTED") or pred_status == "UNCERTAIN_IMAGE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot generate an agronomic disease report for an image that is not verified as a valid cotton leaf.",
        )

    if prediction.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this report.",
        )

    reports_dir = get_settings().resolve_artifacts_dir() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.utcnow().strftime("%Y%m%d")
    pdf_filename = f"cottonguard_report_{prediction_id}_{date_tag}.pdf"
    pdf_path = reports_dir / pdf_filename

    generate_pdf_report(prediction, pdf_path)
    size_bytes = pdf_path.stat().st_size

    report_row = (
        db.query(ReportRecord)
        .filter(ReportRecord.prediction_id == prediction_id, ReportRecord.user_id == current_user.id)
        .first()
    )
    if report_row:
        report_row.pdf_filename = pdf_filename
        report_row.file_size_bytes = size_bytes
        report_row.pdf_path = str(pdf_path)
    else:
        report_row = ReportRecord(
            prediction_id=prediction_id,
            user_id=current_user.id,
            pdf_filename=pdf_filename,
            pdf_path=str(pdf_path),
            file_size_bytes=size_bytes,
        )
        db.add(report_row)

    record_audit_event(
        db,
        action="PDF_REPORT_DOWNLOAD",
        resource=f"report:{prediction_id}",
        details=f"Generated {pdf_filename} ({size_bytes} bytes)",
        user=current_user,
    )
    db.commit()
    db.refresh(report_row)
    return report_row


@router.post("/{prediction_id}/generate")
def generate_report_endpoint(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Generates (or refreshes) the PDF report on disk and returns metadata."""
    row = _ensure_report_file(prediction_id=prediction_id, current_user=current_user, db=db)
    return {
        "status": "generated",
        "prediction_id": prediction_id,
        "report_id": row.id,
        "pdf_filename": row.pdf_filename,
        "file_size_bytes": row.file_size_bytes,
        "download_url": f"/api/reports/{prediction_id}",
    }


@router.get("/{prediction_id}")
def download_pdf_report(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generates, logs in DB, and streams the diagnostic PDF report."""
    row = _ensure_report_file(prediction_id=prediction_id, current_user=current_user, db=db)
    return FileResponse(
        path=row.pdf_path,
        media_type="application/pdf",
        filename=row.pdf_filename,
    )
