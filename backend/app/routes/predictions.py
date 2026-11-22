"""
Prediction API routes: Synchronous & Real-Time SSE Streaming Leaf Analysis and Retrieval.
"""

from typing import Optional, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from backend.app.auth import get_current_user
from backend.app.database import get_db
from backend.app.models import Prediction, PredictionJob, User
from backend.app.schemas import (
    PaginatedHistoryResponse,
    PredictionDetailResponse,
    PredictionJobCreateResponse,
    PredictionJobStatusResponse,
)
from backend.app.services.audit_service import record_audit_event
from backend.app.services.job_service import job_manager
from backend.app.services.prediction_service import prediction_service, serialize_prediction_detail

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])


def _start_async_analysis(
    file: UploadFile,
    seg_alpha: float,
    current_user: User,
) -> dict:
    rgb_image, contents, _, ext = prediction_service._decode_upload_image(file)
    job_id = job_manager.enqueue_analysis_job(
        user_id=current_user.id,
        rgb_image=rgb_image,
        raw_bytes=contents,
        ext=ext,
        seg_alpha=max(0.0, min(1.0, seg_alpha)),
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "stream_url": f"/api/predictions/analyze/stream/{job_id}",
        "total_steps": 7,
    }


@router.post(
    "/analyze",
    response_model=Union[PredictionDetailResponse, PredictionJobCreateResponse],
)
def analyze_leaf_image(
    response: Response,
    file: UploadFile = File(...),
    seg_alpha: float = Form(0.40),
    stream: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Uploads and validates a cotton leaf photograph.
    - When `stream=false` (default): runs 7-stage pipeline synchronously and returns PredictionDetailResponse.
    - When `stream=true`: creates an async PredictionJob and returns HTTP 202 + SSE stream URL.
    """
    if stream:
        response.status_code = status.HTTP_202_ACCEPTED
        return _start_async_analysis(file=file, seg_alpha=seg_alpha, current_user=current_user)

    result = prediction_service.process_and_predict(
        file=file,
        user=current_user,
        db=db,
        seg_alpha=max(0.0, min(1.0, seg_alpha)),
    )
    record_audit_event(
        db,
        action="LEAF_SCAN",
        resource=f"prediction:{result['id']}",
        details=f"Class={result['predicted_class']} ({result['confidence']*100:.1f}%), Severity={result['severity_level']}",
        user=current_user,
        commit=True,
    )
    return result


@router.post(
    "/analyze/async",
    response_model=PredictionJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def analyze_leaf_image_async(
    file: UploadFile = File(...),
    seg_alpha: float = Form(0.40),
    current_user: User = Depends(get_current_user),
):
    """Starts an asynchronous prediction job and returns job metadata for SSE subscription."""
    return _start_async_analysis(file=file, seg_alpha=seg_alpha, current_user=current_user)


@router.get("/analyze/stream/{job_id}")
async def stream_analysis_progress(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Authenticated Server-Sent Events (SSE) endpoint streaming real-time pipeline stages."""
    job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "JOB_NOT_FOUND", "message": f"Prediction job '{job_id}' not found."},
        )
    if job.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "You do not have permission to subscribe to this job."},
        )
    return EventSourceResponse(job_manager.subscribe(job_id))


@router.get("/jobs/{job_id}", response_model=PredictionJobStatusResponse)
def get_analysis_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns persisted status of an async inference job."""
    job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "JOB_NOT_FOUND", "message": f"Prediction job '{job_id}' not found."},
        )
    if job.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "You do not have permission to inspect this job."},
        )
    return {
        "job_id": job.id,
        "status": job.status,
        "current_stage": job.current_stage,
        "step": job.step,
        "total_steps": job.total_steps,
        "prediction_id": job.prediction_id,
        "error_code": job.error_code,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }


@router.get("/{prediction_id}", response_model=PredictionDetailResponse)
def get_prediction_detail(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieves full prediction details including detections, masks, Grad-CAM visuals,
    and agronomic recommendations. Enforces strict ownership checks unless admin.
    """
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with ID #{prediction_id} not found.",
        )

    if prediction.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this prediction record.",
        )

    return serialize_prediction_detail(prediction)


@router.delete("/{prediction_id}")
def delete_prediction_endpoint(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Canonical endpoint for DELETE /api/predictions/{id}."""
    from backend.app.routes.history import delete_history_item

    return delete_history_item(
        prediction_id=prediction_id,
        current_user=current_user,
        db=db,
    )


@router.get("", response_model=PaginatedHistoryResponse)
@router.get("/", response_model=PaginatedHistoryResponse)
def list_predictions_endpoint(
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
    """Canonical route for GET /api/predictions returning paginated history."""
    from backend.app.routes.history import get_paginated_history

    return get_paginated_history(
        search=search,
        disease=disease,
        severity=severity,
        start_date=start_date,
        end_date=end_date,
        page=page,
        page_size=page_size,
        current_user=current_user,
        db=db,
    )


@router.get("/{prediction_id}/report")
def get_prediction_pdf_report_endpoint(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Canonical route for GET /api/predictions/{id}/report to download PDF diagnostic report."""
    from backend.app.routes.reports import download_pdf_report

    return download_pdf_report(
        prediction_id=prediction_id,
        current_user=current_user,
        db=db,
    )
