"""
Training CLI for Cotton Leaf YOLO Object Detector (Phase 8).

Strictly adheres to project scientific integrity rules:
- Refuses to train or fabricate bounding-box labels if genuine human-annotated
  detection data is unavailable in data/detection/.
- Enforces test-split leakage check using artifacts/manifests/splits/test.csv.
- Trains CottonLeafYOLODetector (BSD-3/Apache-2.0 permissive pure PyTorch) with
  AMP FP16 optimized for 6 GB RTX 3050 VRAM.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import CottonLeafYOLODetector, build_detector, load_detection_classes
from src.training.detection_losses import YOLODetectionLoss, compute_detection_metrics
from src.utils.paths import resolve_platform_path


class YOLOBoundingBoxDataset(Dataset):
    """Loads verified RGB photographs and normalized YOLO [cls cx cy w h] bounding boxes."""

    def __init__(self, samples: List[Dict[str, Any]], input_size: int = 640):
        self.samples = samples
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rec = self.samples[idx]
        img_path = resolve_platform_path(rec["image_path"])
        lbl_path = resolve_platform_path(rec["label_path"])

        with Image.open(img_path) as img:
            rgb = img.convert("RGB").resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)

        boxes: List[List[float]] = []
        labels: List[int] = []
        if lbl_path.is_file():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                cx, cy, bw, bh = [float(v) for v in parts[1:]]
                x1 = max(0.0, (cx - bw / 2.0) * self.input_size)
                y1 = max(0.0, (cy - bh / 2.0) * self.input_size)
                x2 = min(float(self.input_size), (cx + bw / 2.0) * self.input_size)
                y2 = min(float(self.input_size), (cy + bh / 2.0) * self.input_size)
                if (x2 - x1) >= 2.0 and (y2 - y1) >= 2.0:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        }
        return tensor, target


def yolo_collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return imgs, targets


def discover_verified_detection_samples(det_dir: Path) -> List[Dict[str, Any]]:
    images_dir = det_dir / "images"
    labels_dir = det_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return []

    test_csv = PROJECT_ROOT / "artifacts" / "manifests" / "splits" / "test.csv"
    test_names = set()
    if test_csv.is_file():
        df_t = pd.read_csv(test_csv)
        if "relative_path" in df_t.columns:
            test_names.update(Path(p).name for p in df_t["relative_path"].dropna().astype(str))

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    valid_samples: List[Dict[str, Any]] = []
    for img_p in sorted(images_dir.iterdir()):
        if img_p.suffix.lower() not in exts:
            continue
        if img_p.name in test_names:
            continue
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        if not lbl_p.is_file():
            continue
        lines = [ln for ln in lbl_p.read_text(encoding="utf-8").splitlines() if len(ln.strip().split()) == 5]
        if lines:
            valid_samples.append({"image_path": str(img_p), "label_path": str(lbl_p), "box_count": len(lines)})
    return valid_samples


def main():
    parser = argparse.ArgumentParser(description="Train Permissive Cotton Leaf YOLO Object Detector")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-dir", type=str, default="data/detection")
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints/detection")
    args = parser.parse_args()

    det_dir = PROJECT_ROOT / args.det_dir
    samples = discover_verified_detection_samples(det_dir)

    if len(samples) < 10:
        print("\n" + "!" * 80)
        print("[DETECTION PREREQUISITE NOTICE]")
        print("YOLO training requires at least 10 genuine human-verified annotated images")
        print("(with verified bounding boxes converted from CVAT Diseased_Region polygons).")
        print(f"Currently found: {len(samples)} valid non-test annotated images in '{det_dir}'.")
        print("\nRun `python scripts/prepare_cvat_annotations.py --input <cvat_export.zip>` once CVAT annotations are ready.")
        print("Status: YOLO training safely skipped (no synthetic or fabricated boxes generated).")
        print("!" * 80 + "\n")
        sys.exit(0)

    random.seed(42)
    random.shuffle(samples)
    val_count = max(1, int(len(samples) * 0.2))
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    train_ds = YOLOBoundingBoxDataset(train_samples, input_size=args.imgsz)
    val_ds = YOLOBoundingBoxDataset(val_samples, input_size=args.imgsz)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=yolo_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=yolo_collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = load_detection_classes()
    model = build_detector(num_classes=len(class_names), input_size=args.imgsz, device=device)
    criterion = YOLODetectionLoss(num_classes=len(class_names))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    ckpt_dir = PROJECT_ROOT / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_yolo.pt"
    best_map50 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outs = model(imgs)
                loss = criterion(outs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * imgs.size(0)

        avg_loss = running_loss / max(1, len(train_ds))

        # Validation pass
        model.eval()
        val_preds: List[List[Dict[str, Any]]] = []
        val_gts: List[Dict[str, Any]] = []
        for s_rec, (_, tgt) in zip(val_samples, val_loader):
            with Image.open(resolve_platform_path(s_rec["image_path"])) as pil_img:
                rgb_resized = pil_img.convert("RGB").resize((args.imgsz, args.imgsz), Image.Resampling.BILINEAR)
            preds = model.predict(rgb_resized, conf_threshold=0.20, iou_threshold=0.45)
            val_preds.append(preds)
            val_gts.append({"boxes": tgt[0]["boxes"].numpy(), "labels": tgt[0]["labels"].numpy()})

        metrics = compute_detection_metrics(val_preds, val_gts, num_classes=len(class_names))
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f} | "
            f"Val P: {metrics['precision']:.4f} R: {metrics['recall']:.4f} "
            f"mAP50: {metrics['mAP50']:.4f} mAP50-95: {metrics['mAP50_95']:.4f}"
        )

        if metrics["mAP50"] >= best_map50:
            best_map50 = metrics["mAP50"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "input_size": args.imgsz,
                "metrics": metrics,
            }, best_path)

    print(f"\n[COMPLETE] Best YOLO checkpoint saved to {best_path}")


if __name__ == "__main__":
    main()
