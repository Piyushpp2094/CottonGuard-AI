"""
Unified AI Inference & Prediction Service for Cotton Leaf AI Backend.

Orchestrates:
  - Image upload security & Pillow validation
  - Model A: EfficientNet-B0 7-class classification (active checkpoint)
  - Model B: Permissive Anchor-Free YOLO region detection (if trained checkpoint present)
  - Model C: U-Net 3-class semantic segmentation (if trained checkpoint present)
  - Model D: Grad-CAM explainability saliency visualizations
  - Affected area pixel quantification & Severity estimation
  - Offline ICAR-CICR agronomic recommendation generation
  - Database persistence across Users, Predictions, Detection, Segmentation & Severity
"""

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi import HTTPException, UploadFile, status
from PIL import Image
from scipy import ndimage
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from backend.app.config import get_settings
from backend.app.models import (
    DetectionResult,
    Prediction,
    PredictionResult,
    SegmentationResult,
    SeverityResult,
    User,
)
from backend.app.utils.image_validation import read_and_validate_upload_file
from src.data.dataset import CLASS_NAMES
from src.data.detection_dataset import FOREGROUND_CLASSES
from src.data.preprocessing import get_inference_transforms
from src.explainability.detection_visualization import draw_bounding_boxes
from src.explainability.gradcam import GradCAM
from src.explainability.visualization import SCIENTIFIC_DISCLAIMER, save_explanation_artifacts
from src.metrics.area_calculation import calculate_affected_area_from_mask, generate_segmentation_overlays
from src.metrics.severity import SeverityEstimator
from src.models.detector import build_detector, draw_detection_boxes
from src.models.faster_rcnn import build_faster_rcnn_model
from src.models.model_factory import build_classifier
from src.models.unet import build_unet
from src.models.unet_resnet34 import build_unet_resnet34
from src.recommendations.recommendation_engine import RecommendationEngine
from src.utils.paths import resolve_platform_path
from src.validation.image_quality import ImageQualityValidator
from src.models.verifier import VerificationService
import torchvision.transforms.functional as TF


def serialize_prediction_detail(prediction: Prediction) -> Dict[str, Any]:
    """Converts a persisted Prediction ORM object (and relations) into PredictionDetailResponse."""
    res = prediction.result
    det = prediction.detection
    seg = prediction.segmentation
    sev = prediction.severity

    recommendations = {}
    if res and res.recommendations_json:
        try:
            recommendations = json.loads(res.recommendations_json)
        except json.JSONDecodeError:
            recommendations = {}

    all_probs = None
    if res and res.all_probabilities_json:
        try:
            all_probs = json.loads(res.all_probabilities_json)
        except json.JSONDecodeError:
            all_probs = None

    # Attempt to retrieve verified image dimensions for historical records lacking normalized_bbox
    img_dims: Optional[Tuple[int, int]] = None
    try:
        cand_path = None
        if prediction.image_path:
            p = resolve_platform_path(prediction.image_path)
            if p.is_file():
                cand_path = p
        if not cand_path and prediction.image_filename:
            p = get_settings().resolve_artifacts_dir() / "predictions" / prediction.image_filename
            if p.is_file():
                cand_path = p
        if cand_path and cand_path.is_file():
            with Image.open(cand_path) as img:
                w, h = img.size
                if w > 0 and h > 0:
                    img_dims = (w, h)
    except Exception:
        img_dims = None

    detections_list: List[Dict[str, Any]] = []
    if det and det.detections_json:
        try:
            raw_dets = json.loads(det.detections_json)
            if isinstance(raw_dets, list):
                for item in raw_dets:
                    if isinstance(item, dict):
                        b = item.get("bbox") or item.get("box") or [0.0, 0.0, 0.0, 0.0]
                        cid = item.get("class_id") if item.get("class_id") is not None else item.get("label", 0)

                        # Resolve class name consistently with the 6 foreground classes
                        cname = item.get("class_name")
                        if not cname or cname == "Unknown":
                            cid_int = int(cid)
                            if 1 <= cid_int <= len(FOREGROUND_CLASSES):
                                cname = FOREGROUND_CLASSES[cid_int - 1]
                            elif 0 <= cid_int < len(FOREGROUND_CLASSES):
                                cname = FOREGROUND_CLASSES[cid_int]
                            else:
                                cname = "Unknown"

                        # Handle normalized_bbox: only compute if verified image dimensions are available
                        nb = item.get("normalized_bbox")
                        if nb is not None and isinstance(nb, list) and len(nb) == 4:
                            norm_bbox = [round(float(x), 4) for x in nb]
                        elif img_dims is not None and len(b) == 4:
                            w, h = img_dims
                            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                            norm_bbox = [
                                round(((x1 + x2) / 2.0) / max(w, 1), 4),
                                round(((y1 + y2) / 2.0) / max(h, 1), 4),
                                round((x2 - x1) / max(w, 1), 4),
                                round((y2 - y1) / max(h, 1), 4),
                            ]
                        else:
                            norm_bbox = []

                        conf = float(item.get("confidence", 0.0))
                        detections_list.append({
                            "bbox": [round(float(x), 1) for x in b],
                            "normalized_bbox": norm_bbox,
                            "class_id": int(cid),
                            "class_name": str(cname),
                            "confidence": round(float(conf), 4),
                            "box": [round(float(x), 1) for x in b],
                            "label": int(cid),
                        })
        except json.JSONDecodeError:
            detections_list = []

    image_filename = prediction.image_filename or resolve_platform_path(prediction.image_path).name
    image_url = f"/artifacts/predictions/{image_filename}"

    return {
        "id": prediction.id,
        "user_id": prediction.user_id,
        "image_url": image_url,
        "predicted_class": prediction.predicted_class,
        "confidence": round(float(prediction.confidence), 4),
        "all_probabilities": all_probs,
        "affected_area_percentage": prediction.affected_area_percentage,
        "severity_level": prediction.severity_level,
        "severity_label": sev.severity_label if sev else None,
        "thresholds_verified": bool(sev.thresholds_verified) if sev else False,
        "model_version": prediction.model_version,
        "status": prediction.status,
        "created_at": prediction.created_at,
        "latency_ms": round(float(res.latency_ms), 2) if res and res.latency_ms is not None else None,
        # Detection
        "detection_status": det.status if det else "UNAVAILABLE",
        "detection_model_version": det.model_version if det else "UNAVAILABLE",
        "detection_count": det.detection_count if det else 0,
        "detections": detections_list,
        "detection_image_url": det.annotated_image_path if det else None,
        "detection_notes": det.notes if det else "YOLO lesion localization checkpoint is not trained yet (awaiting verified bounding-box annotations).",
        # Segmentation
        "segmentation_status": seg.status if seg else ("SUCCESS" if res and res.overlay_path else "UNAVAILABLE"),
        "segmentation_model_version": seg.model_version if seg else "UNAVAILABLE",
        "leaf_pixels": seg.total_leaf_pixels if seg else None,
        "disease_pixels": seg.disease_pixels if seg else None,
        "healthy_pixels": seg.healthy_pixels if seg else None,
        "overlay_url": (seg.overlay_path if seg else None) or (res.overlay_path if res else None),
        "leaf_mask_url": (seg.leaf_mask_path if seg else None) or (res.leaf_mask_path if res else None),
        "disease_mask_url": (seg.disease_mask_path if seg else None) or (res.disease_mask_path if res else None),
        # Grad-CAM
        "gradcam_heatmap_url": res.gradcam_heatmap_path if res else None,
        "gradcam_overlay_url": res.gradcam_overlay_path if res else None,
        "gradcam_comparison_url": res.gradcam_comparison_path if res else None,
        # Advisories
        "recommendations": recommendations,
        "uncertainty_notes": res.uncertainty_notes if res else None,
        "inference_device": None,
        "warning": None,
        # Cotton Leaf Verification Gate & Quality fields
        "is_cotton_leaf": (getattr(prediction, "is_cotton_leaf", None) if isinstance(getattr(prediction, "is_cotton_leaf", None), (bool, type(None))) else True),
        "verification_status": (getattr(prediction, "verification_status", None) if isinstance(getattr(prediction, "verification_status", None), str) else "COTTON_LEAF_VERIFIED"),
        "verification_confidence": (float(getattr(prediction, "verification_confidence", None)) if isinstance(getattr(prediction, "verification_confidence", None), (float, int)) else None),
        "rejection_reason": (getattr(prediction, "rejection_reason", None) if isinstance(getattr(prediction, "rejection_reason", None), str) else None),
        "user_message": (getattr(prediction, "rejection_reason", None) if isinstance(getattr(prediction, "rejection_reason", None), str) else ("Valid cotton leaf verified." if getattr(prediction, "is_cotton_leaf", True) else "Not a Cotton Leaf")),
        "quality_metrics": None,
        "probabilities": all_probs,
    }


class PredictionService:
    """Manages model lifecycle and executes GPU-efficient inference pipelines."""

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._gpu_lock = threading.RLock()
        self.classifier = None
        self.faster_rcnn = None
        self.yolo_detector = None
        self.unet = None
        self.cam_explainer: Optional[GradCAM] = None

        self.severity_estimator = SeverityEstimator()
        self.recommendation_engine = RecommendationEngine()
        self.quality_validator = ImageQualityValidator()

        self.verifier_ckpt = PROJECT_ROOT / "artifacts" / "checkpoints" / "verification" / "best_verifier.pt"
        self.verifier_service = VerificationService(
            checkpoint_path=self.verifier_ckpt,
            device=self.device,
            verification_threshold=0.70,
            uncertainty_lower_bound=0.45,
        )

        self.cls_transform = get_inference_transforms(image_size=224)
        self.seg_transform = get_inference_transforms(image_size=256)

        self.classifier_ckpt = PROJECT_ROOT / "artifacts" / "checkpoints" / "classification" / "best_model.pt"
        self.faster_rcnn_ckpt = PROJECT_ROOT / "artifacts" / "checkpoints" / "detection" / "best_faster_rcnn.pt"
        self.yolo_ckpt = PROJECT_ROOT / "artifacts" / "checkpoints" / "detection" / "best_yolo.pt"
        self.unet_ckpt = PROJECT_ROOT / "artifacts" / "checkpoints" / "segmentation" / "best_unet.pt"

        self._load_classifier()
        self._load_faster_rcnn_if_available()
        self._load_yolo_if_available()
        self._load_unet_if_available()

    @property
    def predictions_dir(self) -> Path:
        p = get_settings().resolve_artifacts_dir() / "predictions"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def explainability_dir(self) -> Path:
        p = get_settings().resolve_artifacts_dir() / "explainability"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _load_classifier(self) -> None:
        if self.classifier_ckpt.is_file():
            print(f"[PREDICTION SERVICE] Loading EfficientNet-B0 classifier from {self.classifier_ckpt} on {self.device}...")
            self.classifier = build_classifier(
                model_name="efficientnet_b0",
                num_classes=len(CLASS_NAMES),
                pretrained=False,
                checkpoint_path=self.classifier_ckpt,
                device=self.device,
            )
            self.classifier.eval()
            self.cam_explainer = GradCAM(model=self.classifier, device=self.device)
        else:
            print(f"[PREDICTION SERVICE] Warning: Classifier checkpoint missing at {self.classifier_ckpt}")

    def _load_faster_rcnn_if_available(self) -> None:
        if self.faster_rcnn_ckpt.is_file():
            print(f"[PREDICTION SERVICE] Loading Faster R-CNN detector from {self.faster_rcnn_ckpt} on {self.device}...")
            try:
                self.faster_rcnn = build_faster_rcnn_model(
                    num_classes=7,
                    pretrained=False,
                    checkpoint_path=self.faster_rcnn_ckpt,
                    device=self.device,
                )
                self.faster_rcnn.eval()
                print(f"[PREDICTION SERVICE] Faster R-CNN ResNet-50 FPN V2 loaded successfully (Epoch 13, mAP50=13.94%, Experimental).")
            except Exception as e_frcnn:
                print(f"[PREDICTION SERVICE] Warning: Failed to load Faster R-CNN checkpoint: {e_frcnn}")
                self.faster_rcnn = None
        else:
            self.faster_rcnn = None

    def _load_yolo_if_available(self) -> None:
        if self.yolo_ckpt.is_file():
            print(f"[PREDICTION SERVICE] Loading YOLO detector from {self.yolo_ckpt}...")
            ckpt = torch.load(self.yolo_ckpt, map_location=self.device)
            input_sz = int(ckpt.get("input_size", 640)) if isinstance(ckpt, dict) else 640
            self.yolo_detector = build_detector(
                input_size=input_sz,
                checkpoint_path=self.yolo_ckpt,
                device=self.device,
            )
            self.yolo_detector.eval()
        else:
            self.yolo_detector = None

    def _load_unet_if_available(self) -> None:
        if self.unet_ckpt.is_file():
            print(f"[PREDICTION SERVICE] Loading U-Net segmentation model from {self.unet_ckpt}...")
            try:
                # Try UNet-ResNet34 (active Phase 4/8 architecture)
                self.unet = build_unet_resnet34(num_classes=3, checkpoint_path=self.unet_ckpt, device=self.device)
                self.unet.eval()
                print(f"[PREDICTION SERVICE] U-Net ResNet-34 loaded successfully.")
            except Exception as e_resnet:
                try:
                    # Fallback to CottonLeafUNet
                    self.unet = build_unet(num_classes=3, checkpoint_path=self.unet_ckpt, device=self.device)
                    self.unet.eval()
                    print(f"[PREDICTION SERVICE] CottonLeafUNet loaded successfully.")
                except Exception as e_unet:
                    print(f"[PREDICTION SERVICE] Warning: Failed to load U-Net checkpoint from {self.unet_ckpt}: {e_resnet} | {e_unet}")
                    self.unet = None
        else:
            self.unet = None

    def get_model_readiness(self) -> Dict[str, Any]:
        """Returns live model checkpoint availability, GPU telemetry, and readiness summary."""
        vram_info: Dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_info.update({
                "device_name": torch.cuda.get_device_name(0),
                "total_vram_mb": round(props.total_memory / (1024 ** 2), 1),
                "allocated_vram_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1),
                "reserved_vram_mb": round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1),
            })

        return {
            "hardware": vram_info,
            "models": {
                "verification": {
                    "name": "Cotton Leaf Verifier (EfficientNet-B0)",
                    "ready": self.verifier_service is not None and self.verifier_service.model is not None,
                    "checkpoint_exists": self.verifier_ckpt.is_file(),
                    "version": "CottonVerifier-EfficientNetB0-v1.0",
                    "status": "trained",
                    "test_accuracy": 0.9984,
                    "test_far": 0.0000,
                    "decision_threshold": 0.70,
                },
                "classification": {
                    "name": "EfficientNet-B0",
                    "ready": self.classifier is not None,
                    "checkpoint_exists": self.classifier_ckpt.is_file(),
                    "version": "EfficientNet-B0-v1.0",
                    "status": "trained",
                    "verified_test_accuracy": 0.9782,
                },
                "detection": {
                    "name": "Faster R-CNN ResNet-50 FPN V2 (Experimental)",
                    "ready": self.faster_rcnn is not None or self.yolo_detector is not None,
                    "checkpoint_exists": self.faster_rcnn_ckpt.is_file() or self.yolo_ckpt.is_file(),
                    "version": "FasterRCNN-ResNet50-FPN-V2-v1.0 (Experimental)" if self.faster_rcnn else ("YOLO-AnchorFree-v1.0" if self.yolo_detector else "UNAVAILABLE"),
                    "status": "experimental" if self.faster_rcnn else ("trained" if self.yolo_detector else "not_trained"),
                    "map50": 0.1394 if self.faster_rcnn else None,
                    "map50_95": 0.0853 if self.faster_rcnn else None,
                    "reason": None if (self.faster_rcnn or self.yolo_detector) else "Requires genuine CVAT bounding-box annotations.",
                },
                "segmentation": {
                    "name": "UNet-ResNet34 (3-Class)",
                    "ready": self.unet is not None,
                    "checkpoint_exists": self.unet_ckpt.is_file(),
                    "version": "UNet-ResNet34-v1.0 (Experimental)" if self.unet else "UNAVAILABLE",
                    "status": "experimental" if self.unet else "not_trained",
                    "dice": 0.6845 if self.unet else None,
                    "iou": 0.5935 if self.unet else None,
                    "reason": None if self.unet else "Requires genuine CVAT pixel masks in data/annotations/.",
                },
                "explainability": {
                    "name": "Grad-CAM (features[-1])",
                    "ready": self.cam_explainer is not None,
                    "role": "Interpretability aid only (not segmentation)",
                },
            },
        }

    def _decode_upload_image(self, file: UploadFile) -> Tuple[Image.Image, bytes, str, str]:
        """Validates uploaded file extension, magic bytes, size <= 10 MB, and resolution."""
        return read_and_validate_upload_file(file)

    def _run_classifier_inference(
        self,
        rgb_image: Image.Image,
    ) -> Tuple[str, float, Dict[str, float], str, Optional[str]]:
        if self.classifier is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "MODEL_UNAVAILABLE", "message": "Classification checkpoint (best_model.pt) is not loaded."},
            )
        tensor_cpu = self.cls_transform(rgb_image).unsqueeze(0)
        used_device = str(self.device)
        warning_msg: Optional[str] = None

        with self._gpu_lock:
            try:
                tensor = tensor_cpu.to(self.device)
                with torch.inference_mode():
                    logits = self.classifier(tensor)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cpu_device = torch.device("cpu")
                self.classifier.to(cpu_device)
                try:
                    with torch.inference_mode():
                        logits = self.classifier(tensor_cpu.to(cpu_device))
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                finally:
                    if self.device.type == "cuda":
                        self.classifier.to(self.device)
                used_device = "cpu"
                warning_msg = "CUDA out of memory; retried classification safely on CPU fallback."

        pred_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        all_probs = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}
        return predicted_class, confidence, all_probs, used_device, warning_msg

    def _run_detection(
        self,
        rgb_image: Image.Image,
        base_name: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        save_visual: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes disease bounding box detection using Faster R-CNN (primary experimental)
        or YOLO (fallback if trained).
        """
        if self.faster_rcnn is not None:
            img_tensor = TF.to_tensor(rgb_image)
            warning_msg = None
            with self._gpu_lock:
                try:
                    with torch.inference_mode():
                        preds = self.faster_rcnn([img_tensor.to(self.device)])
                except torch.cuda.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    cpu_dev = torch.device("cpu")
                    self.faster_rcnn.to(cpu_dev)
                    try:
                        with torch.inference_mode():
                            preds = self.faster_rcnn([img_tensor.to(cpu_dev)])
                    finally:
                        if self.device.type == "cuda":
                            self.faster_rcnn.to(self.device)
                    warning_msg = "CUDA out of memory; retried Faster R-CNN detection safely on CPU fallback."

            p_dict = preds[0]
            p_boxes = p_dict["boxes"].cpu().numpy()
            p_labels = p_dict["labels"].cpu().numpy()
            p_scores = p_dict["scores"].cpu().numpy()

            mask = p_scores >= conf_threshold
            filtered_boxes = p_boxes[mask]
            filtered_labels = p_labels[mask]
            filtered_scores = p_scores[mask]

            orig_w, orig_h = rgb_image.size
            detections: List[Dict[str, Any]] = []
            for box, lbl, sc in zip(filtered_boxes, filtered_labels, filtered_scores):
                lbl_int = int(lbl)
                cls_name = FOREGROUND_CLASSES[lbl_int - 1] if 1 <= lbl_int <= len(FOREGROUND_CLASSES) else "Unknown"
                x1 = float(np.clip(box[0], 0.0, orig_w))
                y1 = float(np.clip(box[1], 0.0, orig_h))
                x2 = float(np.clip(box[2], 0.0, orig_w))
                y2 = float(np.clip(box[3], 0.0, orig_h))
                cx_norm = round(((x1 + x2) / 2.0) / max(orig_w, 1), 4)
                cy_norm = round(((y1 + y2) / 2.0) / max(orig_h, 1), 4)
                w_norm = round((x2 - x1) / max(orig_w, 1), 4)
                h_norm = round((y2 - y1) / max(orig_h, 1), 4)
                detections.append({
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "normalized_bbox": [cx_norm, cy_norm, w_norm, h_norm],
                    "class_id": lbl_int,
                    "class_name": cls_name,
                    "confidence": round(float(sc), 4),
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "label": lbl_int,
                })

            annotated_url = None
            if save_visual and len(filtered_boxes) > 0:
                annotated_img = draw_bounding_boxes(
                    image=rgb_image,
                    boxes=filtered_boxes,
                    labels=[int(l) - 1 for l in filtered_labels],
                    scores=filtered_scores,
                    score_threshold=conf_threshold,
                )
                det_name = f"{base_name}_frcnn_det.jpg"
                annotated_img.save(self.predictions_dir / det_name, quality=92)
                annotated_url = f"/artifacts/predictions/{det_name}"

            msg = (
                f"Detected {len(detections)} lesion region(s) (Faster R-CNN ResNet-50 FPN V2, threshold={conf_threshold})."
                if detections
                else f"No symptomatic bounding boxes exceeded confidence threshold ({conf_threshold})."
            )
            return {
                "status": "EXPERIMENTAL",
                "available": True,
                "model_version": "FasterRCNN-ResNet50-FPN-V2-v1.0 (Experimental, mAP50=13.94%)",
                "detection_count": len(detections),
                "detections": detections,
                "annotated_image_url": annotated_url,
                "message": msg,
                "map50": 0.1394,
                "map50_95": 0.0853,
                "warning": warning_msg,
            }

        if self.yolo_detector is not None:
            detections = self.yolo_detector.predict(
                rgb_image,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            annotated_url = None
            if save_visual:
                annotated_img = draw_detection_boxes(rgb_image, detections)
                det_name = f"{base_name}_yolo_det.jpg"
                annotated_img.save(self.predictions_dir / det_name, quality=92)
                annotated_url = f"/artifacts/predictions/{det_name}"

            return {
                "status": "SUCCESS",
                "available": True,
                "model_version": "YOLO-AnchorFree-v1.0",
                "detection_count": len(detections),
                "detections": detections,
                "annotated_image_url": annotated_url,
                "message": f"Detected {len(detections)} lesion region(s)." if detections else "No symptomatic bounding boxes exceeded threshold.",
            }

        return {
            "status": "UNAVAILABLE",
            "available": False,
            "model_version": "UNAVAILABLE",
            "detection_count": 0,
            "detections": [],
            "annotated_image_url": None,
            "message": (
                "Object detection checkpoint is unavailable. "
                "Verified CVAT bounding-box annotations are required before lesion localization becomes active."
            ),
        }

    def _run_yolo_detection(self, *args, **kwargs) -> Dict[str, Any]:
        """Backward-compatibility alias for _run_detection."""
        return self._run_detection(*args, **kwargs)

    def _post_process_segmentation_mask(self, raw_mask: np.ndarray, min_disease_cc_size: int = 50) -> np.ndarray:
        """
        Morphological post-processing for U-Net predictions:
        1. Hole filling on leaf mask to eliminate small intra-leaf voids.
        2. Small connected component suppression on diseased regions (< 50 px noise).
        3. Strict containment enforcement: diseased regions must reside within leaf mask.
        """
        if raw_mask.ndim != 2:
            return raw_mask
        cleaned = raw_mask.copy().astype(np.uint8)
        leaf_binary = (cleaned > 0).astype(np.uint8)

        # Hole filling for leaf boundary
        leaf_filled = ndimage.binary_fill_holes(leaf_binary).astype(np.uint8)

        # Filter small noise components in diseased region
        disease_binary = (cleaned == 2).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(disease_binary, connectivity=8)

        filtered_disease = np.zeros_like(disease_binary)
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area >= min_disease_cc_size:
                filtered_disease[labels == lbl] = 1

        # Enforce containment: diseased tissue strictly inside leaf
        filtered_disease = (filtered_disease == 1) & (leaf_filled == 1)

        refined = np.zeros_like(raw_mask, dtype=np.uint8)
        refined[leaf_filled == 1] = 1
        refined[filtered_disease] = 2
        return refined

    def _run_unet_segmentation(
        self,
        rgb_image: Image.Image,
        base_name: str,
        alpha: float = 0.40,
    ) -> Dict[str, Any]:
        if self.unet is None:
            return {
                "status": "UNAVAILABLE",
                "available": False,
                "model_version": "UNAVAILABLE",
                "affected_area_percentage": None,
                "leaf_pixels": None,
                "disease_pixels": None,
                "healthy_pixels": None,
                "overlay_url": None,
                "leaf_mask_url": None,
                "disease_mask_url": None,
                "message": (
                    "U-Net segmentation checkpoint (best_unet.pt) is unavailable. "
                    "Genuine pixel-level annotations from CVAT are required to train U-Net and compute real leaf/disease masks."
                ),
            }

        # Fixed 256x256 preprocessing transform aligned with U-Net training
        seg_tensor = self.seg_transform(rgb_image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            seg_logits = self.unet(seg_tensor)
            seg_mask = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()

        # Upsample discrete mask to original image resolution before pixel quantification/visualization
        orig_w, orig_h = rgb_image.size
        if seg_mask.shape != (orig_h, orig_w):
            mask_resized = Image.fromarray(seg_mask.astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.NEAREST)
            seg_mask = np.array(mask_resized, dtype=np.uint8)

        # Apply morphological cleanup and strict containment
        seg_mask = self._post_process_segmentation_mask(seg_mask)

        area_metrics = calculate_affected_area_from_mask(seg_mask)
        overlays = generate_segmentation_overlays(rgb_image, seg_mask, alpha=alpha)

        overlay_name = f"{base_name}_overlay.jpg"
        leaf_mask_name = f"{base_name}_leaf_mask.png"
        disease_mask_name = f"{base_name}_disease_mask.png"

        overlays["overlay"].save(self.predictions_dir / overlay_name, quality=92)
        overlays["leaf_mask"].save(self.predictions_dir / leaf_mask_name)
        overlays["disease_mask"].save(self.predictions_dir / disease_mask_name)

        return {
            "status": "SUCCESS" if area_metrics["is_valid"] else area_metrics["status"],
            "available": True,
            "model_version": "U-Net-3Class-v1.0",
            "affected_area_percentage": area_metrics["affected_area_percentage"],
            "leaf_pixels": area_metrics["leaf_pixels"],
            "disease_pixels": area_metrics["disease_pixels"],
            "healthy_pixels": area_metrics["healthy_pixels"],
            "overlay_url": f"/artifacts/predictions/{overlay_name}",
            "leaf_mask_url": f"/artifacts/predictions/{leaf_mask_name}",
            "disease_mask_url": f"/artifacts/predictions/{disease_mask_name}",
            "message": area_metrics["message"],
        }

    def _gate_check(self, rgb_image: Image.Image) -> None:
        """Enforces image quality validation and cotton leaf verification gates."""
        q_res = self.quality_validator.validate(rgb_image)
        if not q_res.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image quality insufficient: {q_res.user_message}",
            )
        v_res = self.verifier_service.verify(rgb_image)
        if v_res.is_cotton_leaf is not True:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Verification gate failed ({v_res.status}): {v_res.user_message}",
            )

    def classify_uploaded_image(self, file: UploadFile) -> Dict[str, Any]:
        t0 = time.perf_counter()
        rgb_image, _, _, _ = self._decode_upload_image(file)
        self._gate_check(rgb_image)
        pred_class, confidence, all_probs, used_device, warning_msg = self._run_classifier_inference(rgb_image)
        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs,
            "model_version": "EfficientNet-B0-v1.0",
            "device": used_device,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "warning": warning_msg,
        }

    def detect_uploaded_image(
        self,
        file: UploadFile,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Dict[str, Any]:
        rgb_image, _, _, _ = self._decode_upload_image(file)
        self._gate_check(rgb_image)
        base_name = f"det_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        return self._run_yolo_detection(rgb_image, base_name, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

    def segment_uploaded_image(self, file: UploadFile, alpha: float = 0.40) -> Dict[str, Any]:
        rgb_image, _, _, _ = self._decode_upload_image(file)
        self._gate_check(rgb_image)
        base_name = f"seg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        return self._run_unet_segmentation(rgb_image, base_name, alpha=alpha)

    def process_validated_image(
        self,
        *,
        rgb_image: Image.Image,
        contents: bytes,
        ext: str,
        user: User,
        db: Session,
        seg_alpha: float = 0.40,
        progress_callback: Optional[Callable[[str, str, int, int, Optional[Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Any]:
        """Executes the full 7-stage diagnostic pipeline with VRAM lock & CPU fallback."""
        def emit(stage: str, stage_status: str, step: int, summary: Optional[Dict[str, Any]] = None) -> None:
            if progress_callback:
                progress_callback(stage, stage_status, step, 7, summary)

        t0 = time.perf_counter()
        unique_id = uuid.uuid4().hex[:10]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_base_name = f"{timestamp}_{unique_id}"
        image_save_name = f"{safe_base_name}{ext}"
        image_save_path = self.predictions_dir / image_save_name

        with open(image_save_path, "wb") as f:
            f.write(contents)

        # Stage 1: Image Quality Validation
        emit("quality_validation", "running", 1)
        quality_res = self.quality_validator.validate(rgb_image)
        if not quality_res.is_valid:
            emit("quality_validation", "rejected", 1, {"reason": quality_res.user_message})
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            db_prediction = Prediction(
                user_id=user.id,
                image_filename=image_save_name,
                image_path=str(image_save_path),
                predicted_class="Invalid Image Quality",
                confidence=0.0,
                affected_area_percentage=None,
                severity_level="UNAVAILABLE",
                model_version="ImageQuality-v1.0",
                status="REJECTED_QUALITY",
                is_cotton_leaf=False,
                verification_status="REJECTED_QUALITY",
                verification_confidence=0.0,
                rejection_reason=quality_res.user_message,
            )
            db.add(db_prediction)
            db.flush()

            no_rec = {
                "immediate_first_steps": [],
                "preventive_management": [],
                "critical_warning": None,
                "summary": quality_res.user_message,
            }
            db_result = PredictionResult(
                prediction_id=db_prediction.id,
                recommendations_json=json.dumps(no_rec),
                uncertainty_notes=quality_res.user_message,
                latency_ms=latency_ms,
            )
            db.add(db_result)

            db_det = DetectionResult(
                prediction_id=db_prediction.id,
                detections_json="[]",
                detection_count=0,
                model_version="UNAVAILABLE",
                status="REJECTED",
                notes="Image quality validation failed; detection skipped.",
            )
            db.add(db_det)

            db_seg = SegmentationResult(
                prediction_id=db_prediction.id,
                model_version="UNAVAILABLE",
                status="REJECTED",
                notes="Image quality validation failed; segmentation skipped.",
            )
            db.add(db_seg)

            db_sev = SeverityResult(
                prediction_id=db_prediction.id,
                severity_level="UNAVAILABLE",
                is_available=False,
                thresholds_verified=False,
                notes=quality_res.user_message,
            )
            db.add(db_sev)

            db.commit()
            db.refresh(db_prediction)

            detail = serialize_prediction_detail(db_prediction)
            detail["is_cotton_leaf"] = False
            detail["verification_status"] = "REJECTED_QUALITY"
            detail["verification_confidence"] = 0.0
            detail["rejection_reason"] = quality_res.user_message
            detail["user_message"] = quality_res.user_message
            detail["quality_metrics"] = quality_res.metrics
            emit("quality_validation", "complete", 7, {"prediction_id": db_prediction.id, "rejected": True})
            return detail

        emit("quality_validation", "complete", 1, quality_res.metrics)

        # Stage 2: Cotton Leaf Domain Verification Gate
        emit("cotton_leaf_verification", "running", 2)
        with self._gpu_lock:
            v_res = self.verifier_service.verify(rgb_image)

        if v_res.is_cotton_leaf is False or v_res.is_cotton_leaf is None:
            # Mandatory Gate: Reject non-cotton / non-plant / ambiguous inputs!
            emit("cotton_leaf_verification", "rejected" if v_res.is_cotton_leaf is False else "uncertain", 2, {
                "status": v_res.status,
                "confidence": v_res.confidence,
                "reason": v_res.rejection_reason,
            })
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            pred_label = "Not a Cotton Leaf" if v_res.is_cotton_leaf is False else "Verification Uncertain"

            db_prediction = Prediction(
                user_id=user.id,
                image_filename=image_save_name,
                image_path=str(image_save_path),
                predicted_class=pred_label,
                confidence=v_res.confidence,
                affected_area_percentage=None,
                severity_level="UNAVAILABLE",
                model_version="CottonVerifier-EfficientNetB0-v1.0",
                status=v_res.status,
                is_cotton_leaf=v_res.is_cotton_leaf,
                verification_status=v_res.status,
                verification_confidence=v_res.confidence,
                rejection_reason=v_res.rejection_reason,
            )
            db.add(db_prediction)
            db.flush()

            no_rec = {
                "immediate_first_steps": [],
                "preventive_management": [],
                "critical_warning": None,
                "summary": v_res.user_message,
            }
            db_result = PredictionResult(
                prediction_id=db_prediction.id,
                all_probabilities_json=json.dumps(v_res.probabilities),
                recommendations_json=json.dumps(no_rec),
                uncertainty_notes=v_res.rejection_reason,
                latency_ms=latency_ms,
            )
            db.add(db_result)

            db_det = DetectionResult(
                prediction_id=db_prediction.id,
                detections_json="[]",
                detection_count=0,
                model_version="UNAVAILABLE",
                status="REJECTED",
                notes=f"Verification gate active ({v_res.status}); detection skipped.",
            )
            db.add(db_det)

            db_seg = SegmentationResult(
                prediction_id=db_prediction.id,
                model_version="UNAVAILABLE",
                status="REJECTED",
                notes=f"Verification gate active ({v_res.status}); segmentation skipped.",
            )
            db.add(db_seg)

            db_sev = SeverityResult(
                prediction_id=db_prediction.id,
                severity_level="UNAVAILABLE",
                is_available=False,
                thresholds_verified=False,
                notes=v_res.rejection_reason,
            )
            db.add(db_sev)

            db.commit()
            db.refresh(db_prediction)

            detail = serialize_prediction_detail(db_prediction)
            detail["is_cotton_leaf"] = v_res.is_cotton_leaf
            detail["verification_status"] = v_res.status
            detail["verification_confidence"] = v_res.confidence
            detail["rejection_reason"] = v_res.rejection_reason
            detail["user_message"] = v_res.user_message
            detail["probabilities"] = v_res.probabilities
            emit("cotton_leaf_verification", "complete", 7, {"prediction_id": db_prediction.id, "rejected": True})
            return detail

        emit("cotton_leaf_verification", "complete", 2, {
            "status": "COTTON_LEAF_VERIFIED",
            "confidence": v_res.confidence,
        })

        # Stage 3: Classification (EfficientNet-B0) - ONLY FOR CONFIRMED COTTON LEAF
        emit("classification", "running", 3)
        predicted_class, confidence, all_probs, used_device, oom_warning = self._run_classifier_inference(rgb_image)
        emit(
            "classification",
            "complete",
            3,
            {"predicted_class": predicted_class, "confidence": round(confidence, 4), "device": used_device},
        )

        # Stage 4: Grad-CAM Explainability
        emit("gradcam", "running", 4)
        gradcam_paths: Optional[Dict[str, str]] = None
        if self.cam_explainer is not None:
            try:
                with self._gpu_lock:
                    cam_res = self.cam_explainer.generate_cam(rgb_image, target_size=rgb_image.size)
                gradcam_paths = save_explanation_artifacts(
                    original_image=rgb_image,
                    heatmap=cam_res["heatmap"],
                    metadata=cam_res,
                    base_name=safe_base_name,
                    output_dir=self.explainability_dir,
                    alpha=0.4,
                    colormap="jet",
                )
            except Exception as exc:
                print(f"[PREDICTION SERVICE] Warning: Grad-CAM failed: {exc}")
        emit("gradcam", "complete", 4, {"available": gradcam_paths is not None})

        # Stage 5: YOLO Object Detection (if trained)
        emit("detection", "running", 5)
        with self._gpu_lock:
            det_res = self._run_yolo_detection(rgb_image, safe_base_name)
        emit("detection", "complete", 5, {"status": det_res["status"], "detection_count": det_res["detection_count"]})

        # Stage 6: U-Net Semantic Segmentation (if trained)
        emit("segmentation", "running", 6)
        with self._gpu_lock:
            seg_res = self._run_unet_segmentation(rgb_image, safe_base_name, alpha=seg_alpha)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        emit("segmentation", "complete", 6, {"status": seg_res["status"]})

        # Stage 7: Severity Evaluation & Agronomic Guidance
        emit("severity", "running", 7)
        severity_result = self.severity_estimator.evaluate(
            affected_area_percentage=seg_res["affected_area_percentage"],
            predicted_class=predicted_class,
            confidence=confidence,
            leaf_pixels=seg_res["leaf_pixels"],
            disease_pixels=seg_res["disease_pixels"],
            healthy_pixels=seg_res["healthy_pixels"],
            segmentation_model_version=seg_res["model_version"],
        )
        severity_level = severity_result["level"]

        recommendations = self.recommendation_engine.get_recommendation(
            predicted_class=predicted_class,
            confidence=confidence,
            severity_level=severity_level,
            affected_area_percentage=seg_res["affected_area_percentage"],
        )

        uncertainty_notes = None
        if not seg_res["available"]:
            uncertainty_notes = seg_res["message"]
        elif seg_res["affected_area_percentage"] is None:
            uncertainty_notes = seg_res["message"]

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        db_prediction = Prediction(
            user_id=user.id,
            image_filename=image_save_name,
            image_path=str(image_save_path),
            predicted_class=predicted_class,
            confidence=confidence,
            affected_area_percentage=seg_res["affected_area_percentage"],
            severity_level=severity_level,
            model_version="EfficientNet-B0-v1.0",
            status="SUCCESS",
            is_cotton_leaf=True,
            verification_status="COTTON_LEAF_VERIFIED",
            verification_confidence=v_res.confidence,
            rejection_reason=None,
        )
        db.add(db_prediction)
        db.flush()

        db_result = PredictionResult(
            prediction_id=db_prediction.id,
            leaf_mask_path=seg_res["leaf_mask_url"],
            disease_mask_path=seg_res["disease_mask_url"],
            overlay_path=seg_res["overlay_url"],
            gradcam_heatmap_path=gradcam_paths["heatmap_url"] if gradcam_paths else None,
            gradcam_overlay_path=gradcam_paths["overlay_url"] if gradcam_paths else None,
            gradcam_comparison_path=gradcam_paths["comparison_url"] if gradcam_paths else None,
            gradcam_metadata_path=gradcam_paths["metadata_url"] if gradcam_paths else None,
            all_probabilities_json=json.dumps(all_probs),
            latency_ms=latency_ms,
            recommendations_json=json.dumps(recommendations),
            uncertainty_notes=uncertainty_notes,
        )
        db.add(db_result)

        db_det = DetectionResult(
            prediction_id=db_prediction.id,
            detections_json=json.dumps(det_res["detections"]),
            detection_count=det_res["detection_count"],
            annotated_image_path=det_res["annotated_image_url"],
            model_version=det_res["model_version"],
            status=det_res["status"],
            notes=det_res["message"],
        )
        db.add(db_det)

        db_seg = SegmentationResult(
            prediction_id=db_prediction.id,
            leaf_mask_path=seg_res["leaf_mask_url"],
            disease_mask_path=seg_res["disease_mask_url"],
            overlay_path=seg_res["overlay_url"],
            total_leaf_pixels=seg_res["leaf_pixels"],
            disease_pixels=seg_res["disease_pixels"],
            healthy_pixels=seg_res["healthy_pixels"],
            affected_area_percentage=seg_res["affected_area_percentage"],
            model_version=seg_res["model_version"],
            status=seg_res["status"],
            notes=seg_res["message"],
        )
        db.add(db_seg)

        db_sev = SeverityResult(
            prediction_id=db_prediction.id,
            severity_level=severity_level,
            severity_label=severity_result.get("label"),
            affected_area_percentage=seg_res["affected_area_percentage"],
            is_available=bool(severity_result.get("is_available", False)),
            thresholds_verified=bool(severity_result.get("thresholds_verified", False)),
            notes=severity_result.get("note") or severity_result.get("threshold_basis"),
        )
        db.add(db_sev)

        db.commit()
        db.refresh(db_prediction)

        detail = serialize_prediction_detail(db_prediction)
        detail["inference_device"] = used_device
        detail["warning"] = oom_warning
        emit("recommendations", "complete", 7, {"prediction_id": db_prediction.id})
        return detail

    def process_and_predict(
        self,
        file: UploadFile,
        user: User,
        db: Session,
        base_url: str = "",
        seg_alpha: float = 0.40,
    ) -> Dict[str, Any]:
        """Executes synchronous unified 7-stage inference and database persistence."""
        rgb_image, contents, _, ext = self._decode_upload_image(file)
        return self.process_validated_image(
            rgb_image=rgb_image,
            contents=contents,
            ext=ext,
            user=user,
            db=db,
            seg_alpha=seg_alpha,
        )

    def generate_gradcam_for_image(
        self,
        file: UploadFile,
        target_class: Optional[int] = None,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> Dict[str, Any]:
        """Dedicated Grad-CAM generation for uploaded photographs."""
        if self.cam_explainer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Classifier checkpoint not loaded. Grad-CAM explainer is unavailable.",
            )

        rgb_image, _, _, _ = self._decode_upload_image(file)
        self._gate_check(rgb_image)
        safe_base_name = f"gradcam_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}"

        try:
            cam_res = self.cam_explainer.generate_cam(
                rgb_image,
                target_class=target_class,
                target_size=rgb_image.size,
            )
            paths = save_explanation_artifacts(
                original_image=rgb_image,
                heatmap=cam_res["heatmap"],
                metadata=cam_res,
                base_name=safe_base_name,
                output_dir=self.explainability_dir,
                alpha=alpha,
                colormap=colormap,
            )
            return {
                "predicted_class": cam_res["predicted_class_name"],
                "confidence": round(cam_res["confidence"], 4),
                "target_class": cam_res["target_class_name"],
                "original_url": paths["original_url"],
                "heatmap_url": paths["heatmap_url"],
                "overlay_url": paths["overlay_url"],
                "comparison_url": paths["comparison_url"],
                "metadata_url": paths["metadata_url"],
                "target_layer": cam_res["target_layer_name"],
                "model_architecture": "EfficientNet-B0",
                "device": cam_res["device"],
                "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
                "all_probabilities": {k: round(v, 4) for k, v in cam_res["all_probabilities"].items()},
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Grad-CAM generation error: {exc}",
            )


prediction_service = PredictionService()
