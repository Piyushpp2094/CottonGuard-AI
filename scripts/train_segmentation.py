"""
Training CLI for U-Net Cotton Leaf Semantic Segmentation (Phase 4).

Adheres strictly to the project rules:
- Verifies that authentic image-mask pairs exist in data/annotations/.
- Refuses to fabricate masks or claim training without genuine human-verified annotations.
- Trains U-Net only when at least 10 valid verified mask pairs exist.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet import build_unet
from src.training.segmentation_losses import CombinedDiceCELoss, compute_segmentation_metrics
from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from scripts.validate_annotations import validate_annotation_directory


class PairedSegmentationDataset(Dataset):
    """Dataset loading paired (RGB Image, Class Index Mask) for U-Net."""

    def __init__(self, pairs: List[dict], image_size: int = 256):
        self.pairs = pairs
        self.image_size = image_size
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        img_path = pair["image_path"]
        mask_path = pair["mask_path"]

        with Image.open(img_path) as img:
            img = img.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            img_arr = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)
            img_tensor = (img_tensor - self.mean) / self.std

        with Image.open(mask_path) as mask:
            # Nearest neighbor resize to preserve discrete integer class indices {0, 1, 2}
            mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            mask_arr = np.array(mask, dtype=np.int64)
            # Clip any unexpected value to background (0)
            mask_arr = np.clip(mask_arr, 0, 2)
            mask_tensor = torch.from_numpy(mask_arr)

        return img_tensor, mask_tensor


def main():
    parser = argparse.ArgumentParser(description="Train U-Net Cotton Leaf Segmentation Model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for U-Net training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    parser.add_argument("--image-size", type=int, default=256, help="Image resolution for segmentation")
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints/segmentation")
    args = parser.parse_args()

    images_dir = PROJECT_ROOT / "data" / "annotations" / "images"
    masks_dir = PROJECT_ROOT / "data" / "annotations" / "masks"
    summary_path = PROJECT_ROOT / "artifacts" / "manifests" / "annotation_summary.json"

    # Step 1: Validate annotations
    summary = validate_annotation_directory(images_dir, masks_dir, summary_path)

    if not summary["is_ready_for_segmentation_training"]:
        print("\n" + "!" * 80)
        print("[SEGMENTATION PREREQUISITE NOTICE]")
        print("U-Net training requires at least 10 genuine human-annotated mask pairs.")
        print(f"Currently found: {summary['valid_pairs_count']} valid pairs in '{images_dir.relative_to(PROJECT_ROOT)}'.")
        print("\nTo enable segmentation training:")
        print("  1. Place cotton leaf photographs into: data/annotations/images/")
        print("  2. Place matching 3-class PNG masks into: data/annotations/masks/")
        print("     (0 = Background, 1 = Healthy Leaf, 2 = Diseased Tissue)")
        print("  3. Follow docs/segmentation_annotation_guide.md for CVAT/Label Studio instructions.")
        print("  4. Re-run this training command once annotations are ready.")
        print("!" * 80)
        print("\nStatus: Segmentation training safely skipped (no fabricated data generated).")
        sys.exit(0)

    # If valid annotations exist, proceed to train
    valid_pairs = summary["valid_pairs"]
    np.random.seed(42)
    np.random.shuffle(valid_pairs)

    val_split = max(1, int(len(valid_pairs) * 0.2))
    train_pairs = valid_pairs[val_split:]
    val_pairs = valid_pairs[:val_split]

    train_ds = PairedSegmentationDataset(train_pairs, image_size=args.image_size)
    val_ds = PairedSegmentationDataset(val_pairs, image_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[TRAINING] Initializing U-Net on {device}...")
    model = build_unet(num_classes=3, device=device)
    criterion = CombinedDiceCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_dice = 0.0
    checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "best_unet.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())

        val_loss /= len(val_ds)
        preds_cat = torch.cat(all_preds, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        metrics = compute_segmentation_metrics(preds_cat, targets_cat)

        val_dice = metrics["mean_dice"]
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {metrics['mean_iou']:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            }, best_ckpt_path)
            print(f"--> Saved best U-Net checkpoint to {best_ckpt_path}")

    print("\n[COMPLETE] U-Net training completed successfully.")


if __name__ == "__main__":
    main()
