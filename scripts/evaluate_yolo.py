"""
Evaluation CLI for Cotton Leaf YOLO Object Detector (Phase 8).

Reports Precision, Recall, mAP@0.50, and mAP@0.50:0.95 against verified
human-annotated bounding-box samples. Refuses to run without genuine data & checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import build_detector, load_detection_classes
from src.training.detection_losses import compute_detection_metrics
from scripts.train_yolo import YOLOBoundingBoxDataset, discover_verified_detection_samples
from src.utils.paths import resolve_platform_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Cotton Leaf YOLO Detector")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/detection/best_yolo.pt")
    parser.add_argument("--det-dir", type=str, default="data/detection")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    ckpt_path = PROJECT_ROOT / args.checkpoint
    det_dir = PROJECT_ROOT / args.det_dir
    if not ckpt_path.is_file():
        print(f"[SKIPPED] YOLO checkpoint unavailable at {ckpt_path}. Genuine training required first.")
        sys.exit(0)

    samples = discover_verified_detection_samples(det_dir)
    if not samples:
        print(f"[SKIPPED] No verified detection ground-truth labels in {det_dir}.")
        sys.exit(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = load_detection_classes()
    model = build_detector(num_classes=len(class_names), input_size=args.imgsz, checkpoint_path=ckpt_path, device=device)
    model.eval()

    ds = YOLOBoundingBoxDataset(samples, input_size=args.imgsz)
    all_preds: List[List[Dict[str, Any]]] = []
    all_gts: List[Dict[str, Any]] = []

    for rec, (_, tgt) in zip(samples, ds):
        with Image.open(resolve_platform_path(rec["image_path"])) as pil_img:
            resized = pil_img.convert("RGB").resize((args.imgsz, args.imgsz), Image.Resampling.BILINEAR)
        preds = model.predict(resized, conf_threshold=args.conf, iou_threshold=args.iou)
        all_preds.append(preds)
        all_gts.append({"boxes": tgt["boxes"].numpy(), "labels": tgt["labels"].numpy()})

    metrics = compute_detection_metrics(all_preds, all_gts, num_classes=len(class_names))
    out_path = PROJECT_ROOT / "artifacts" / "metrics" / "yolo_evaluation_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
