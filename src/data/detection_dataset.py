"""
Cotton Leaf Detection Dataset and Collate Utility for Faster R-CNN (Phase 9.2A).

Converts verified published images and normalized YOLO labels into
TorchVision-compatible Faster R-CNN targets:
- Bounding box coordinates in absolute [x1, y1, x2, y2] format.
- Class labels in 1..6 range (1: Bacterial Blight, ..., 6: Leaf Variegation).
- Background label 0 reserved for Faster R-CNN RPN/RoI background.
- Healthy leaf controls have empty targets: boxes shape (0, 4) and labels shape (0,).
- Custom detection_collate_fn handles variable numbers of boxes per image.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTION_DIR = PROJECT_ROOT / "data" / "detection"
MANIFESTS_DIR = PROJECT_ROOT / "artifacts" / "manifests"

FOREGROUND_CLASSES = [
    "Bacterial Blight",
    "Curl Virus",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation",
]

YOLO_TO_TORCHVISION_LABEL = {i: i + 1 for i in range(len(FOREGROUND_CLASSES))}
TORCHVISION_TO_CLASS_NAME = {i + 1: name for i, name in enumerate(FOREGROUND_CLASSES)}
TORCHVISION_TO_CLASS_NAME[0] = "background"


class CottonDetectionDataset(Dataset):
    """
    TorchVision Faster R-CNN Dataset for Cotton Leaf Disease Detection.
    Loads 800x800 RGB images and converts normalized YOLO labels [cls cx cy w h]
    into absolute [x1, y1, x2, y2] pixel coordinates with 1-based class IDs.
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        ignore_dir: Optional[Union[str, Path]] = None,
    ):
        self.split = split.lower()
        self.data_dir = Path(data_dir or DETECTION_DIR)
        self.manifest_path = Path(manifest_path or (MANIFESTS_DIR / "combined_pilot_manifest.csv"))
        self.transform = transform
        self.ignore_dir = Path(ignore_dir) if ignore_dir is not None else (self.data_dir / "ignore_labels")

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        if self.split in ["val", "validation"]:
            self.df = df[df["split"].isin(["val", "validation"])].reset_index(drop=True)
        elif self.split == "train":
            self.df = df[df["split"] == "train"].reset_index(drop=True)
        else:
            raise ValueError(f"Unknown split: {self.split}. Expected 'train' or 'val'/'validation'.")

        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        row = self.df.iloc[idx]
        filename = row["filename"]
        stem = Path(filename).stem

        img_path = self.images_dir / filename
        lbl_path = self.labels_dir / f"{stem}.txt"

        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not lbl_path.is_file():
            raise FileNotFoundError(f"Label file not found: {lbl_path}")

        with Image.open(img_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            w, h = rgb_img.size

        # Convert image to float32 Tensor in [0.0, 1.0]
        img_tensor = TF.to_tensor(rgb_img)

        # Parse labels
        boxes_list: List[List[float]] = []
        labels_list: List[int] = []

        lines = [l.strip() for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            yolo_cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            # Convert normalized center-width-height to absolute pixel coordinates [x1, y1, x2, y2]
            x1 = max(0.0, (cx - bw / 2.0) * w)
            y1 = max(0.0, (cy - bh / 2.0) * h)
            x2 = min(float(w), (cx + bw / 2.0) * w)
            y2 = min(float(h), (cy + bh / 2.0) * h)

            # Ensure positive area
            if x2 > x1 and y2 > y1:
                boxes_list.append([x1, y1, x2, y2])
                # Map YOLO 0..5 -> TorchVision 1..6
                torchvision_label = YOLO_TO_TORCHVISION_LABEL.get(yolo_cls, yolo_cls + 1)
                labels_list.append(torchvision_label)

        if len(boxes_list) > 0:
            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels = torch.as_tensor(labels_list, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Healthy leaf controls: exactly 0 detection targets
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        # Parse optional ignore regions (Tier 2 diffuse/morphological symptoms)
        ignore_boxes_list: List[List[float]] = []
        if self.ignore_dir is not None and self.ignore_dir.is_dir():
            ign_path = self.ignore_dir / f"{stem}.txt"
            if ign_path.is_file():
                ign_lines = [l.strip() for l in ign_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                for line in ign_lines:
                    parts = line.split()
                    if len(parts) == 5:
                        _, icx, icy, ibw, ibh = map(float, parts)
                        ix1 = max(0.0, (icx - ibw / 2.0) * w)
                        iy1 = max(0.0, (icy - ibh / 2.0) * h)
                        ix2 = min(float(w), (icx + ibw / 2.0) * w)
                        iy2 = min(float(h), (icy + ibh / 2.0) * h)
                    elif len(parts) == 4:
                        ix1, iy1, ix2, iy2 = map(float, parts)
                    else:
                        continue
                    if ix2 > ix1 and iy2 > iy1:
                        ignore_boxes_list.append([ix1, iy1, ix2, iy2])

        if len(ignore_boxes_list) > 0:
            ignore_boxes = torch.as_tensor(ignore_boxes_list, dtype=torch.float32)
        else:
            ignore_boxes = torch.zeros((0, 4), dtype=torch.float32)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "ignore_boxes": ignore_boxes,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([h, w], dtype=torch.int64),
        }

        if self.transform is not None:
            img_tensor, target = self.transform(img_tensor, target)

        return img_tensor, target


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], ...]]:
    """
    Collate function for TorchVision Faster R-CNN.
    Returns tuple of images and tuple of targets.
    """
    images, targets = zip(*batch)
    return tuple(images), tuple(targets)
