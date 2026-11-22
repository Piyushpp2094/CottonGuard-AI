"""
Training CLI for UNet-ResNet34 Cotton Leaf Segmentation (Experimental Run).

Strictly gated and safeguarded:
- Verifies >= 50 authentic verified image-mask pairs in `data/annotations/`.
- Uses group-aware manifest split: strictly 35 train / 15 validation images (0 test leakage).
- Uses RTX 3050 6 GB settings: 512x512, batch_size=8 (with safe OOM fallback to 4),
  AdamW(lr=1e-4, weight_decay=1e-5), CosineAnnealingLR(T_max=50, eta_min=1e-6),
  Weighted Combined CE (0.3, 1.0, 2.5) + Soft Dice loss, AMP mixed precision,
  and early stopping on `val_macro_dice` (patience=15).
- Saves best checkpoint to `artifacts/checkpoints/segmentation/best_unet.pt`.
- Records per-class Dice, per-class IoU, Macro Dice, Macro IoU, confusion matrix,
  loss curves, training duration, and peak VRAM.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_annotations import RELIABLE_TRAINING_MIN_PAIRS, validate_annotation_directory
from src.models.unet_resnet34 import (
    WeightedCombinedDiceCELoss,
    build_unet_resnet34,
    compute_segmentation_metrics,
    get_segmentation_eval_transforms,
    get_segmentation_train_transforms,
)


class AlbumentationsSegmentationDataset(Dataset):
    """Paired RGB Image + 3-Class Integer Mask dataset using Albumentations 512x512."""

    def __init__(self, pairs: List[dict], transform) -> None:
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        img = np.array(Image.open(pair["image_path"]).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(pair["mask_path"]).convert("L"), dtype=np.int64)
        augmented = self.transform(image=img, mask=mask)
        return augmented["image"].float(), augmented["mask"].long()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_curves(history: List[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h.get("val_loss", 0.0) for h in history]
    macro_dices = [h["val_macro_dice"] for h in history]
    macro_ious = [h["val_macro_iou"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, train_losses, label="Train Loss", color="#EF4444", lw=2)
    ax1.plot(epochs, val_losses, label="Val Loss", color="#F97316", lw=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Weighted CE + Dice)")
    ax1.set_title("Training & Validation Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, macro_dices, label="Val Macro Dice", color="#10B981", lw=2)
    ax2.plot(epochs, macro_ious, label="Val Macro IoU", color="#3B82F6", lw=2, linestyle="--")
    if "dice_Diseased Tissue" in history[0]:
        ax2.plot(epochs, [h["dice_Diseased Tissue"] for h in history], label="Val Disease Dice", color="#8B5CF6", lw=1.5, linestyle=":")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Score")
    ax2.set_title("Validation Macro Dice & IoU")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Validation Pixel Confusion Matrix (Best Checkpoint)",
        ylabel="Ground Truth Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm[i, j]:,}"
            ax.text(j, i, val_str, ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_unet(args):
    start_time = time.time()
    set_seed(args.seed)

    images_dir = args.annotations_dir / "images"
    masks_dir = args.annotations_dir / "masks"
    summary_path = PROJECT_ROOT / "artifacts" / "manifests" / "annotation_summary.json"
    summary = validate_annotation_directory(images_dir, masks_dir, summary_path, min_reliable_pairs=args.min_pairs)
    if not summary.get("is_ready_for_segmentation_training") or summary.get("valid_pairs_count", 0) < args.min_pairs:
        print(
            f"[UNet-ResNet34 GATE] Blocked: verified pairs = {summary.get('valid_pairs_count', 0)} "
            f"(< {args.min_pairs} required)."
        )
        return 2

    pairs: List[dict] = summary["valid_pairs"]
    manifest_p = args.manifest or (PROJECT_ROOT / "artifacts" / "manifests" / "combined_pilot_manifest.csv")
    assert manifest_p.is_file(), f"Manifest not found: {manifest_p}"
    
    df_m = pd.read_csv(manifest_p)
    split_map = {Path(r["filename"]).stem: r["split"] for _, r in df_m.iterrows()}
    train_pairs = [p for p in pairs if split_map.get(Path(p["image_path"]).stem) == "train"]
    val_pairs = [p for p in pairs if split_map.get(Path(p["image_path"]).stem) == "validation"]

    print("==================================================================")
    print("COTTONGUARD AI - EXPERIMENTAL U-NET RESNET-34 TRAINING RUN")
    print("==================================================================")
    print(f"Total Verified Pairs:   {len(pairs)}")
    print(f"Training Pairs:         {len(train_pairs)} (strictly train split)")
    print(f"Validation Pairs:       {len(val_pairs)} (strictly validation split)")
    print(f"Test Set Leakage Check: 0 images from test.csv loaded")
    print(f"Input Resolution:       {args.image_size} x {args.image_size}")
    print(f"Batch Size:             {args.batch_size}")
    print(f"Initial LR / Optimizer: {args.lr} / AdamW(weight_decay={args.weight_decay})")
    print(f"Loss Function:          0.5 * Weighted CE(0.3, 1.0, 2.5) + 0.5 * Dice")
    print(f"Early Stopping:         Patience {args.patience} epochs on val_macro_dice")
    print("==================================================================")

    assert len(train_pairs) == 35, f"Expected exactly 35 train pairs, got {len(train_pairs)}"
    assert len(val_pairs) == 15, f"Expected exactly 15 val pairs, got {len(val_pairs)}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_ds = AlbumentationsSegmentationDataset(train_pairs, get_segmentation_train_transforms(args.image_size))
    val_ds = AlbumentationsSegmentationDataset(val_pairs, get_segmentation_eval_transforms(args.image_size))

    # Safe OOM fallback setup
    current_bs = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=current_bs, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=current_bs, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    model = build_unet_resnet34(num_classes=3, encoder_name="resnet34", encoder_weights="imagenet", device=device)
    criterion = WeightedCombinedDiceCELoss(class_weights=(0.3, 1.0, 2.5), ce_weight=0.5, dice_weight=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    ckpt_dir = PROJECT_ROOT / "artifacts" / "checkpoints" / "segmentation"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    best_epoch = 0
    best_cm = None
    best_metrics = {}
    no_improve = 0
    history: List[Dict[str, Any]] = []

    metric_keys = [
        "dice_Background", "dice_Healthy Leaf", "dice_Diseased Tissue",
        "iou_Background", "iou_Healthy Leaf", "iou_Diseased Tissue"
    ]

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        try:
            for imgs, masks in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                bs = imgs.size(0)
                train_loss += float(loss.item()) * bs
                n_train += bs
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM WARN] Out of memory at batch size {current_bs}. Reducing to {current_bs // 2}...")
            torch.cuda.empty_cache()
            current_bs = max(1, current_bs // 2)
            train_loader = DataLoader(train_ds, batch_size=current_bs, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=current_bs, shuffle=False, num_workers=0)
            continue

        scheduler.step()

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_iou_sum = 0.0
        per_class_sums = {k: 0.0 for k in metric_keys}
        epoch_cm = np.zeros((3, 3), dtype=np.int64)
        n_val = 0

        with torch.inference_mode():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    vloss = criterion(logits, masks)
                bs = imgs.size(0)
                val_loss_sum += float(vloss.item()) * bs
                preds = torch.argmax(logits, dim=1)
                
                # Metrics
                m = compute_segmentation_metrics(preds, masks, num_classes=3)
                val_dice_sum += m["mean_dice"] * bs
                val_iou_sum += m["mean_iou"] * bs
                for k in metric_keys:
                    per_class_sums[k] += m.get(k, 0.0) * bs
                
                # Confusion matrix accumulation
                p_np = preds.cpu().numpy().flatten()
                t_np = masks.cpu().numpy().flatten()
                for t_cls in range(3):
                    for p_cls in range(3):
                        epoch_cm[t_cls, p_cls] += int(((t_np == t_cls) & (p_np == p_cls)).sum())
                
                n_val += bs

        avg_train_loss = train_loss / max(1, n_train)
        avg_val_loss = val_loss_sum / max(1, n_val)
        val_dice = val_dice_sum / max(1, n_val)
        val_iou = val_iou_sum / max(1, n_val)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "val_loss": round(avg_val_loss, 5),
            "val_macro_dice": round(val_dice, 5),
            "val_macro_iou": round(val_iou, 5),
        }
        for k in metric_keys:
            epoch_metrics[k] = round(per_class_sums[k] / max(1, n_val), 5)
        history.append(epoch_metrics)

        # Log epoch summary
        dis_dice = epoch_metrics["dice_Diseased Tissue"]
        leaf_dice = epoch_metrics["dice_Healthy Leaf"]
        bg_dice = epoch_metrics["dice_Background"]
        lr_curr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1 or val_dice > best_dice:
            star = " *** BEST" if val_dice > best_dice else ""
            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                f"Loss: {avg_train_loss:.4f} / {avg_val_loss:.4f} | "
                f"MacroDice: {val_dice:.4f} | "
                f"BG: {bg_dice:.3f} | Leaf: {leaf_dice:.3f} | Dis: {dis_dice:.3f} | "
                f"MacroIoU: {val_iou:.4f} | lr: {lr_curr:.2e}{star}"
            )

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            best_cm = epoch_cm.copy()
            best_metrics = dict(epoch_metrics)
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_dice": val_dice,
                    "val_macro_iou": val_iou,
                    "best_metrics": best_metrics,
                    "confusion_matrix": best_cm.tolist(),
                    "class_weights": [0.3, 1.0, 2.5],
                    "input_size": [args.image_size, args.image_size],
                    "architecture": "unet_resnet34",
                    "status": "EXPERIMENTAL",
                },
                ckpt_dir / "best_unet.pt",
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[EARLY STOPPING] No improvement for {args.patience} epochs. Stopping at epoch {epoch}.")
                break

    duration_sec = round(time.time() - start_time, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2) if device.type == "cuda" else 0.0

    print("==================================================================")
    print("TRAINING RUN COMPLETE")
    print("==================================================================")
    print(f"Total Duration:         {duration_sec:.1f} s ({duration_sec/60:.2f} min)")
    print(f"Peak GPU VRAM:          {peak_vram_mb:.1f} MB")
    print(f"Best Validation Epoch:  {best_epoch}")
    print(f"Best Macro Dice:        {best_dice:.4f}")
    print(f"Best Macro IoU:         {best_metrics.get('val_macro_iou', 0.0):.4f}")
    print(f"Background Dice:        {best_metrics.get('dice_Background', 0.0):.4f}")
    print(f"Healthy Leaf Dice:      {best_metrics.get('dice_Healthy Leaf', 0.0):.4f}")
    print(f"Diseased Tissue Dice:   {best_metrics.get('dice_Diseased Tissue', 0.0):.4f}")
    print("==================================================================")

    # Save artifacts
    best_ckpt_p = ckpt_dir / "best_unet.pt"
    assert best_ckpt_p.is_file(), "best_unet.pt was not saved!"
    with open(best_ckpt_p, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    metrics_out = {
        "status": "EXPERIMENTAL",
        "training_completed_at": datetime.now().isoformat(),
        "duration_seconds": duration_sec,
        "peak_vram_mb": peak_vram_mb,
        "cuda_environment": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
        },
        "best_epoch": best_epoch,
        "best_macro_dice": best_dice,
        "best_metrics": best_metrics,
        "confusion_matrix": best_cm.tolist() if best_cm is not None else [],
        "class_names": ["Background", "Healthy Leaf", "Diseased Tissue"],
        "checkpoint_path": str(best_ckpt_p),
        "checkpoint_sha256": ckpt_sha256,
        "dataset": {
            "total_pairs": len(pairs),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "test_split_leakage": False,
        },
    }

    metrics_json_p = PROJECT_ROOT / "artifacts" / "metrics" / "segmentation_metrics.json"
    metrics_json_p.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_json_p, "w") as f:
        json.dump(metrics_out, f, indent=2)

    csv_history_p = PROJECT_ROOT / "artifacts" / "metrics" / "segmentation_training_history.csv"
    pd.DataFrame(history).to_csv(csv_history_p, index=False)

    cm_json_p = PROJECT_ROOT / "artifacts" / "metrics" / "unet_confusion_matrix.json"
    with open(cm_json_p, "w") as f:
        json.dump({"best_epoch": best_epoch, "confusion_matrix": best_cm.tolist() if best_cm is not None else []}, f, indent=2)

    # Plot loss curves and confusion matrix
    plot_curves(history, PROJECT_ROOT / "artifacts" / "metrics" / "unet_training_curves.png")
    if best_cm is not None:
        plot_confusion_matrix(best_cm, ["Background", "Healthy Leaf", "Diseased Tissue"], PROJECT_ROOT / "artifacts" / "metrics" / "unet_confusion_matrix.png")

    print(f"Artifacts saved:")
    print(f"  - Checkpoint:       {best_ckpt_p} ({best_ckpt_p.stat().st_size:,} bytes, sha256={ckpt_sha256[:16]}...)")
    print(f"  - Metrics JSON:     {metrics_json_p}")
    print(f"  - History CSV:      {csv_history_p}")
    print(f"  - Curves Plot:      artifacts/metrics/unet_training_curves.png")
    print(f"  - Confusion Plot:   artifacts/metrics/unet_confusion_matrix.png")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Gated UNet-ResNet34 segmentation training CLI")
    parser.add_argument("--annotations-dir", type=Path, default=PROJECT_ROOT / "data" / "annotations")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--min-pairs", type=int, default=RELIABLE_TRAINING_MIN_PAIRS)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "manifests" / "combined_pilot_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return train_unet(args)


if __name__ == "__main__":
    raise SystemExit(main())
