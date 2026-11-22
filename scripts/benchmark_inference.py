"""
Reproducible AI Benchmarking Script for CottonGuard AI (Phase 4).

Preserves scientific integrity rules:
- Keeps `artifacts/manifests/splits/test.csv` UNTOUCHED during routine benchmarking;
  runs live classifier evaluation on `validation.csv` (319 samples) while reporting
  previously audited hold-out test metrics directly from `artifacts/metrics/classification_metrics.json`.
- Uses sequential model loading + `torch.cuda.empty_cache()` to fit within 6 GB RTX 3050 VRAM.
- Reports `mAP50`, `mAP50-95`, `Dice`, and `IoU` honestly as `null` (`NOT_TRAINED`) when
  checkpoints do not exist, and labels untrained UNet-ResNet34 forward pass strictly as
  `untrained_architecture_footprint_only`.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CLASS_NAMES, CottonLeafDataset
from src.data.preprocessing import get_eval_transforms
from src.explainability.gradcam import GradCAM
from src.models.model_factory import build_classifier
from src.models.unet_resnet34 import UNetResNet34Segmentor, build_unet_resnet34
from src.models.yolov8_detector import YOLOv8DiseaseDetector


def _gpu_peak_mb(device: torch.device) -> float | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    idx = device.index if device.index is not None else 0
    return round(torch.cuda.max_memory_allocated(idx) / (1024 ** 2), 2)


def _reset_gpu_stats(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(idx)


def verify_class_names_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.is_file():
        return {"exists": False, "matches_dataset_class_names": False}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = data.get("classes", [])
    return {
        "exists": True,
        "num_classes": len(listed),
        "matches_dataset_class_names": listed == list(CLASS_NAMES),
        "classes": listed,
    }


def benchmark_classifier(
    val_csv: Path,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int = 32,
    use_amp: bool = True,
) -> Dict[str, Any]:
    _reset_gpu_stats(device)
    model = build_classifier(
        model_name="efficientnet_b0",
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model.eval()

    ds = CottonLeafDataset(csv_path=val_csv, transform=get_eval_transforms(224), return_path=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # Warmup pass
    sample_x, _, _ = ds[0]
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
        _ = model(sample_x.unsqueeze(0).to(device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    all_preds: List[int] = []
    all_targets: List[int] = []

    t0 = time.perf_counter()
    with torch.inference_mode():
        for imgs, targets, _paths in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_sec = time.perf_counter() - t0

    # Single-image latency measurement (20 iterations)
    sample_batch = sample_x.unsqueeze(0).to(device)
    latencies_ms: List[float] = []
    with torch.inference_mode():
        for _ in range(20):
            ts = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                _ = model(sample_batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies_ms.append((time.perf_counter() - ts) * 1000.0)

    peak_vram = _gpu_peak_mb(device)

    # Grad-CAM benchmark on first validation sample
    _, _, first_img_path = ds[0]
    sample_pil = Image.open(first_img_path).convert("RGB")
    _reset_gpu_stats(device)
    cam = GradCAM(model=model, device=device)
    t_cam0 = time.perf_counter()
    _ = cam.generate_cam(sample_pil, target_size=sample_pil.size)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cam_ms = round((time.perf_counter() - t_cam0) * 1000.0, 2)
    cam_vram = _gpu_peak_mb(device)

    del cam, model
    _reset_gpu_stats(device)

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(CLASS_NAMES))))
    return {
        "split_evaluated": str(val_csv),
        "test_set_untouched": True,
        "samples_evaluated": len(all_targets),
        "device": str(device),
        "amp_enabled": bool(use_amp and device.type == "cuda"),
        "batch_size": batch_size,
        "validation_metrics": {
            "accuracy": round(float(accuracy_score(all_targets, all_preds)), 4),
            "macro_precision": round(float(precision_score(all_targets, all_preds, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(all_targets, all_preds, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(all_targets, all_preds, average="macro", zero_division=0)), 4),
            "confusion_matrix": cm.tolist(),
        },
        "batch_throughput_imgs_per_sec": round(len(all_targets) / max(1e-6, total_sec), 2),
        "single_image_mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "single_image_p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "peak_gpu_memory_mb": peak_vram,
        "gradcam_single_image_latency_ms": cam_ms,
        "gradcam_peak_gpu_memory_mb": cam_vram,
    }


def benchmark_untrained_unet_footprint(device: torch.device) -> Dict[str, Any]:
    """Measures architectural 512x512 UNet-ResNet34 memory/latency without claiming trained accuracy."""
    _reset_gpu_stats(device)
    model = build_unet_resnet34(num_classes=3, encoder_name="resnet34", encoder_weights=None, device=device)
    model.eval()
    dummy = torch.randn(1, 3, 512, 512, device=device)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    latencies: List[float] = []
    with torch.inference_mode():
        for _ in range(10):
            ts = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - ts) * 1000.0)

    peak_vram = _gpu_peak_mb(device)
    del model, dummy
    _reset_gpu_stats(device)

    return {
        "footprint_scope": "untrained_architecture_footprint_only",
        "input_resolution": [512, 512],
        "single_image_mean_latency_ms": round(float(np.mean(latencies)), 2),
        "peak_gpu_memory_mb": peak_vram,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproducible CottonGuard AI inference benchmark")
    parser.add_argument("--val-csv", type=Path, default=PROJECT_ROOT / "artifacts" / "manifests" / "splits" / "validation.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "metrics" / "benchmark_report.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cls_ckpt = PROJECT_ROOT / "artifacts" / "checkpoints" / "classification" / "best_model.pt"
    cls_test_metrics_path = PROJECT_ROOT / "artifacts" / "metrics" / "classification_metrics.json"
    class_manifest_path = PROJECT_ROOT / "artifacts" / "manifests" / "class_names.json"

    audited_test_metrics = {}
    if cls_test_metrics_path.is_file():
        raw_tm = json.loads(cls_test_metrics_path.read_text(encoding="utf-8"))
        audited_test_metrics = {
            "source": str(cls_test_metrics_path),
            "overall_performance": raw_tm.get("overall_performance", {}),
            "confusion_matrix": raw_tm.get("confusion_matrix", []),
        }

    cls_bench = benchmark_classifier(args.val_csv, cls_ckpt, device=device, batch_size=args.batch_size, use_amp=True)
    yolo_wrapper = YOLOv8DiseaseDetector()
    unet_wrapper = UNetResNet34Segmentor()
    unet_footprint = benchmark_untrained_unet_footprint(device=device)

    report = {
        "benchmark_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1) if device.type == "cuda" else None,
        },
        "class_names_verification": verify_class_names_manifest(class_manifest_path),
        "models": {
            "classifier_efficientnet_b0": {
                "status": "trained",
                "live_validation_benchmark": cls_bench,
                "pre_audited_holdout_test_metrics": audited_test_metrics,
            },
            "detector_yolov8n": {
                "status": yolo_wrapper.status,
                "map50": None,
                "map50_95": None,
                "inference_time_ms": None,
                "peak_gpu_memory_mb": None,
                "metadata": yolo_wrapper.get_metadata(),
            },
            "segmentor_unet_resnet34": {
                "status": unet_wrapper.status,
                "dice": None,
                "iou": None,
                "metadata": unet_wrapper.get_metadata(),
                "architecture_forward_footprint": unet_footprint,
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
