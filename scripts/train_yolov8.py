"""
Training CLI for Ultralytics YOLOv8n Cotton Leaf Disease Detector (Phase 4/6).

Strictly gated:
- Refuses to train or fabricate labels if `datasets/yolo/data.yaml` or verified
  YOLO bounding-box `.txt` label files are missing.
- Checks `ultralytics` installation (AGPL-3.0) explicitly before training.
"""

import argparse
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.yolov8_detector import DEFAULT_YOLO_CONFIG, is_ultralytics_installed


def verify_yolo_dataset(data_yaml: Path) -> int:
    if not data_yaml.is_file():
        return 0
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    base = Path(cfg.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    train_rel = cfg.get("train", "images/train")
    train_img_dir = base / train_rel
    if not train_img_dir.is_dir():
        return 0
    return sum(1 for p in train_img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Gated Ultralytics YOLOv8n training CLI")
    parser.add_argument("--config", type=Path, default=DEFAULT_YOLO_CONFIG)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "datasets" / "yolo" / "data.yaml")
    args = parser.parse_args()

    train_count = verify_yolo_dataset(args.data)
    if train_count == 0:
        print(
            "[YOLOv8n GATE] Training blocked: no verified YOLO dataset/images found at "
            f"{args.data}. Export CVAT bounding-box annotations first (no synthetic boxes allowed)."
        )
        return 2

    if not is_ultralytics_installed():
        print(
            "[YOLOv8n GATE] 'ultralytics' (AGPL-3.0) is not installed in this environment. "
            "Review AGPL-3.0 obligations and install `ultralytics` before running training."
        )
        return 3

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from ultralytics import YOLO  # type: ignore

    model = YOLO(cfg.get("model", "yolov8n.pt"))
    model.train(
        data=str(args.data),
        epochs=int(cfg.get("epochs", 100)),
        imgsz=int(cfg.get("imgsz", 640)),
        batch=int(cfg.get("batch", 16)),
        optimizer=str(cfg.get("optimizer", "AdamW")),
        lr0=float(cfg.get("lr0", 0.001)),
        lrf=float(cfg.get("lrf", 0.01)),
        warmup_epochs=int(cfg.get("warmup_epochs", 3)),
        patience=int(cfg.get("patience", 20)),
        device=cfg.get("device", 0),
        workers=int(cfg.get("workers", 4)),
        seed=int(cfg.get("seed", 42)),
        project=str(PROJECT_ROOT / cfg.get("project", "artifacts/yolo_runs")),
        name=str(cfg.get("name", "cotton_disease_v1")),
        save_period=int(cfg.get("save_period", 10)),
        augment=bool(cfg.get("augment", True)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
