"""
Explainability API routes: Grad-CAM Visual Heatmaps, Overlays and Diagnostics.
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import GradCAMResponse
from backend.app.auth import get_current_user
from backend.app.services.prediction_service import prediction_service

router = APIRouter(prefix="/api/explainability", tags=["Explainability"])


@router.post("/gradcam", response_model=GradCAMResponse)
def generate_gradcam_explanation(
    file: UploadFile = File(...),
    target_class: Optional[int] = Form(None),
    alpha: float = Form(0.4),
    colormap: str = Form("jet"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM) for a cotton leaf image.
    
    Highlights salient foliar regions contributing to the predicted disease class
    (or an explicitly requested diagnostic target class).
    Returns URLs to the generated heatmap, blended overlay, and three-panel comparison.
    """
    result = prediction_service.generate_gradcam_for_image(
        file=file,
        target_class=target_class,
        alpha=alpha,
        colormap=colormap
    )
    return result
