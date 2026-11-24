"""
FastAPI Main Application for CottonGuard AI Platform.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.auth import get_current_user
from backend.app.config import PROJECT_ROOT, get_settings
from backend.app.database import get_db, init_db
from backend.app.models import User
from backend.app.routes import (
    admin,
    ai_models,
    authentication,
    dashboard,
    explainability,
    history,
    predictions,
    reports,
)
from backend.app.services.model_registry import build_and_save_model_registry, read_model_registry
from backend.app.services.prediction_service import prediction_service

ARTIFACTS_ROOT = get_settings().resolve_artifacts_dir()
PREDICTIONS_DIR = ARTIFACTS_ROOT / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
EXPLAINABILITY_DIR = ARTIFACTS_ROOT / "explainability"
EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initializes database tables, applies migrations, and synchronizes model registry."""
    init_db()
    build_and_save_model_registry()
    print("[APP] CottonGuard AI database & model registry initialized.")
    yield


app = FastAPI(
    title="CottonGuard AI — Leaf Disease Analysis & Severity Estimation Platform",
    description=(
        "Full-stack local AI backend integrating EfficientNet-B0 classification, "
        "YOLO lesion detection, U-Net semantic segmentation, Grad-CAM explainability, "
        "real-time SSE progress streaming, severity analysis, ICAR-CICR advisories, and PDF reports."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        code = str(exc.detail.get("code", f"HTTP_{exc.status_code}"))
        message = str(exc.detail.get("message", "Request error."))
        extra_detail = exc.detail.get("detail")
        compat_detail = message
    else:
        code = f"HTTP_{exc.status_code}"
        message = str(exc.detail)
        extra_detail = None
        compat_detail = exc.detail

    return JSONResponse(
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
        content={
            "detail": compat_detail,
            "error": {
                "code": code,
                "message": message,
                "detail": extra_detail,
            },
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "detail": exc.errors(),
            },
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, _exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error.",
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred.",
                "detail": None,
            },
        },
    )


# CORS configuration allowing Vite frontend (:5173), backend static (:8000), HuggingFace (:7860), and any cloud origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:7860",
        "http://127.0.0.1:7860",
    ],
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/artifacts/predictions", StaticFiles(directory=str(PREDICTIONS_DIR)), name="predictions_static")
app.mount("/artifacts/explainability", StaticFiles(directory=str(EXPLAINABILITY_DIR)), name="explainability_static")
app.mount("/artifacts/metrics", StaticFiles(directory=str(METRICS_DIR)), name="metrics_static")

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static_web")

app.include_router(authentication.router)
app.include_router(history.router)
app.include_router(predictions.router)
app.include_router(dashboard.router)
app.include_router(reports.router)
app.include_router(explainability.router)
app.include_router(ai_models.router)
app.include_router(admin.router)


@app.get("/")
@app.head("/")
def root():
    """Serves the fallback web app if present or platform metadata JSON."""
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return {
        "platform": "CottonGuard AI Precision Agriculture Platform",
        "status": "Online",
        "version": "2.1.0",
        "documentation": "/docs",
        "frontend_dev_url": "http://localhost:5173",
    }


@app.get("/app")
@app.head("/app")
def serve_app():
    """Direct route to the built-in static dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail={"code": "STATIC_MISSING", "message": "Static fallback UI absent."})
    return FileResponse(str(index_file))


@app.post("/api/predict", tags=["Predictions"])
def predict_sync_alias(
    response: Response,
    file: UploadFile = File(...),
    seg_alpha: float = Form(0.40),
    stream: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Canonical alias routing directly to /api/predictions/analyze."""
    from backend.app.routes.predictions import analyze_leaf_image

    return analyze_leaf_image(
        response=response,
        file=file,
        seg_alpha=seg_alpha,
        stream=stream,
        current_user=current_user,
        db=db,
    )


@app.get("/api/health")
@app.head("/api/health")
def health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Returns live GPU/CUDA telemetry, DB connectivity, and readiness for all AI models."""
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    readiness = prediction_service.get_model_readiness()
    return {
        "status": "healthy" if db_ok and readiness["models"]["classification"]["ready"] else "degraded",
        "database_connected": db_ok,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "classification_checkpoint": readiness["models"]["classification"]["checkpoint_exists"],
        "detection_checkpoint": readiness["models"]["detection"]["checkpoint_exists"],
        "segmentation_checkpoint": readiness["models"]["segmentation"]["checkpoint_exists"],
        "hardware": readiness["hardware"],
        "models": readiness["models"],
    }


@app.get("/api/health/models")
def health_models() -> Dict[str, Any]:
    """Returns the persisted model registry (`artifacts/model_registry.json`) and live runtime telemetry."""
    registry = read_model_registry()
    readiness = prediction_service.get_model_readiness()
    return {
        "registry": registry,
        "hardware": readiness["hardware"],
        "models": registry.get("models", {}),
    }
