"""
Evaluation CLI for U-Net Cotton Leaf Semantic Segmentation (Phase 7).

Evaluates Dice Coefficient, IoU, Precision, Recall, and Pixel Accuracy on
validated 3-class image-mask pairs when checkpoint exists.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet import build_unet
from src.training.segmentation_losses import compute_segmentation_metrics
from scripts.train_segmentation import PairedSegmentationDataset
from scripts.validate_annotations import validate_annotation_directory


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cotton Leaf U-Net Segmentation Model")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/segmentation/best_unet.pt")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    ckpt_path = PROJECT_ROOT / args.checkpoint
    if not ckpt_path.is_file():
        print(f"[SKIPPED] U-Net checkpoint unavailable at {ckpt_path}. Genuine training required first.")
        sys.exit(0)

    summary = validate_annotation_directory(
        PROJECT_ROOT / "data" / "annotations" / "images",
        PROJECT_ROOT / "data" / "annotations" / "masks",
        PROJECT_ROOT / "artifacts" / "manifests" / "annotation_summary.json",
    )
    if summary["valid_pairs_count"] == 0:
        print("[SKIPPED] Zero verified segmentation mask pairs available for evaluation.")
        sys.exit(0)

    ds = PairedSegmentationDataset(summary["valid_pairs"], image_size=args.image_size)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_unet(num_classes=3, checkpoint_path=ckpt_path, device=device)
    model.eval()

    preds_list, targets_list = [], []
    with torch.inference_mode():
        for imgs, masks in loader:
            logits = model(imgs.to(device))
            preds_list.append(torch.argmax(logits, dim=1).cpu())
            targets_list.append(masks)

    metrics = compute_segmentation_metrics(torch.cat(preds_list, dim=0), torch.cat(targets_list, dim=0), num_classes=3)
    out_path = PROJECT_ROOT / "artifacts" / "metrics" / "unet_evaluation_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
