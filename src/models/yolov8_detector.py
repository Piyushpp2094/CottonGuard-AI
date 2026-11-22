"""
Separate Ultralytics YOLOv8n Disease Region Detector Wrapper (Phase 4).

License Notice:
  The `ultralytics` runtime library is distributed under GNU AGPL-3.0.
  This wrapper uses lazy imports so the rest of the CottonGuard AI platform
  (including the permissive BSD-3/Apache-2.0 custom `CottonLeafYOLODetector` in
  `src/models/detector.py`) continues operating cleanly until/unless AGPL-3.0
  obligations are explicitly accepted and `ultralytics` is installed.

Status Policy:
  Remains strictly `NOT_TRAINED` until a real checkpoint trained on verified
  CVAT bounding-box annotations exists.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_CONFIG = PROJECT_ROOT / "configs" / "yolo_config.yaml"
DEFAULT_YOLO_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "detection" / "best_yolo.pt"

YOLOV8_DETECTION_CLASSES: List[str] = [
    "Bacterial_Blight",
    "Curl_Virus",
    "Herbicide_Growth_Damage",
    "Leaf_Hopper_Jassids",
    "Leaf_Redding",
    "Leaf_Variegation",
]


def is_ultralytics_installed() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("ultralytics") is not None
    except Exception:
        return False


class YOLOv8DiseaseDetector:
    """
    Gated YOLOv8n wrapper for 6-class symptomatic cotton leaf region localization.
    Healthy Leaf samples must produce zero false-positive diseased boxes.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_YOLO_CHECKPOINT
        self.config_path = Path(config_path) if config_path else DEFAULT_YOLO_CONFIG
        self.device = device or ("0" if torch.cuda.is_available() else "cpu")
        self._model = None

    @property
    def is_trained(self) -> bool:
        return self.checkpoint_path.is_file()

    @property
    def status(self) -> str:
        return "trained" if self.is_trained else "NOT_TRAINED"

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "YOLOv8n",
            "architecture": "yolov8n",
            "license": "AGPL-3.0 (ultralytics)",
            "ultralytics_installed": is_ultralytics_installed(),
            "status": self.status,
            "checkpoint_path": str(self.checkpoint_path) if self.is_trained else None,
            "config_path": str(self.config_path),
            "classes": list(YOLOV8_DETECTION_CLASSES),
            "annotation_requirements": "CVAT bounding-box annotations exported in YOLO format required before training.",
        }

    def _load_model(self):
        if not self.is_trained:
            raise FileNotFoundError(f"YOLOv8n checkpoint not trained yet at: {self.checkpoint_path}")
        if not is_ultralytics_installed():
            raise RuntimeError(
                "Package 'ultralytics' (AGPL-3.0) is not installed in the virtual environment. "
                "Approve AGPL-3.0 licensing terms and install `ultralytics` before running YOLOv8n inference."
            )
        if self._model is None:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(str(self.checkpoint_path))
        return self._model

    def predict(
        self,
        image: Union[Image.Image, str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Dict[str, Any]:
        """Runs YOLOv8n detection if trained, otherwise returns honest NOT_TRAINED payload."""
        if not self.is_trained:
            return {
                "status": "NOT_TRAINED",
                "available": False,
                "model_name": "YOLOv8n",
                "model_version": None,
                "detection_count": 0,
                "detections": [],
                "reason": (
                    "YOLOv8n checkpoint is not trained yet. Genuine CVAT bounding-box annotations "
                    "are required before lesion detection becomes active."
                ),
            }

        model = self._load_model()
        used_device = self.device
        warning = None
        try:
            results = model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=640,
                device=used_device,
                verbose=False,
            )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            used_device = "cpu"
            warning = "CUDA out of memory during YOLOv8n inference; retried safely on CPU."
            results = model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=640,
                device="cpu",
                verbose=False,
            )

        detections: List[Dict[str, Any]] = []
        if results and results[0].boxes is not None:
            r0 = results[0]
            h, w = r0.orig_shape
            for box in r0.boxes:
                xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0].item())
                cls_name = (
                    YOLOV8_DETECTION_CLASSES[cls_id]
                    if 0 <= cls_id < len(YOLOV8_DETECTION_CLASSES)
                    else str(cls_id)
                )
                detections.append(
                    {
                        "bbox": xyxy,
                        "normalized_bbox": [
                            round(xyxy[0] / max(1, w), 4),
                            round(xyxy[1] / max(1, h), 4),
                            round(xyxy[2] / max(1, w), 4),
                            round(xyxy[3] / max(1, h), 4),
                        ],
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(float(box.conf[0].item()), 4),
                    }
                )

        return {
            "status": "success",
            "available": True,
            "model_name": "YOLOv8n",
            "model_version": "v1.0.0",
            "inference_device": used_device,
            "warning": warning,
            "detection_count": len(detections),
            "detections": detections,
        }
