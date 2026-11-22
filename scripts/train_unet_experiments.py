"""
Controlled Experimental Runner for U-Net ResNet-34 Refinement (Phase 8.1).

Runs single-variable controlled experiments on identical 35-train / 15-val split:
- Exp 2: Resolution 768x768 (batch size 4, Weighted CE + Dice)
- Exp 3: Focal Loss + Soft Dice (512x512, gamma=2.0, alpha=[0.3, 1.0, 2.5])
- Exp 4a: Disease weight 1.5 ([0.3, 1.0, 1.5], 512x512, Weighted CE + Dice)
- Exp 4b: Disease weight 2.0 ([0.3, 1.0, 2.0], 512x512, Weighted CE + Dice)
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet_resnet34 import (
    WeightedCombinedDiceCELoss,
    build_unet_resnet34,
    compute_segmentation_metrics,
    get_segmentation_eval_transforms,
    get_segmentation_train_transforms,
)
from src.training.segmentation_losses import SoftDiceLoss


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with class weights alpha and focusing parameter gamma."""

    def __init__(self, alpha: Tuple[float, float, float] = (0.3, 1.0, 2.5), gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), targets: (B, H, W)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # (B, H, W)
        pt = torch.exp(-ce_loss)  # probability of true class
        alpha_weights = self.alpha[targets]  # (B, H, W)
        focal_loss = alpha_weights * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class CombinedFocalDiceLoss(nn.Module):
    """0.5 * FocalLoss + 0.5 * SoftDiceLoss."""

    def __init__(self, alpha: Tuple[float, float, float] = (0.3, 1.0, 2.5), gamma: float = 2.0) -> None:
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.focal(logits, targets) + 0.5 * self.dice(logits, targets)


class AlbumentationsSegmentationDataset(Dataset):
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


def run_experiment(exp_name: str, image_size: int, batch_size: int, loss_type: str, class_weights: Tuple[float, float, float], epochs: int = 60, patience: int = 15, seed: int = 42, force_retrain: bool = False) -> Dict[str, Any]:
    set_seed(seed)
    manifest_p = PROJECT_ROOT / "artifacts" / "manifests" / "combined_pilot_manifest.csv"
    seg_img_dir = PROJECT_ROOT / "data" / "annotations" / "images"
    seg_mask_dir = PROJECT_ROOT / "data" / "annotations" / "masks"
    ckpt_dir = PROJECT_ROOT / "artifacts" / "checkpoints" / "segmentation"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = ckpt_dir / f"{exp_name}_best_unet.pt"

    df_m = pd.read_csv(manifest_p)
    train_df = df_m[df_m["split"] == "train"]
    val_df = df_m[df_m["split"] == "validation"]

    train_pairs = [{"image_path": seg_img_dir / r["filename"], "mask_path": seg_mask_dir / f"{Path(r['filename']).stem}.png"} for _, r in train_df.iterrows()]
    val_pairs = [{"image_path": seg_img_dir / r["filename"], "mask_path": seg_mask_dir / f"{Path(r['filename']).stem}.png"} for _, r in val_df.iterrows()]

    print(f"\n=======================================================")
    print(f"RUNNING: {exp_name.upper()}")
    print(f"Size: {image_size}x{image_size} | Batch: {batch_size} | Loss: {loss_type} | Weights: {class_weights}")
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Max Epochs: {epochs} | Patience: {patience}")
    print(f"=======================================================")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_ds = AlbumentationsSegmentationDataset(train_pairs, get_segmentation_train_transforms(image_size))
    val_ds = AlbumentationsSegmentationDataset(val_pairs, get_segmentation_eval_transforms(image_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    model = build_unet_resnet34(num_classes=3, encoder_name="resnet34", encoder_weights="imagenet", device=device)

    if loss_type == "focal":
        criterion = CombinedFocalDiceLoss(alpha=class_weights, gamma=2.0).to(device)
    else:
        criterion = WeightedCombinedDiceCELoss(class_weights=class_weights, ce_weight=0.5, dice_weight=0.5).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

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

    t0 = time.time()
    skip_training = out_ckpt.is_file() and not force_retrain

    if skip_training:
        print(f"[{exp_name}] Existing checkpoint found at {out_ckpt}. Skipping training loop and loading saved weights...")
        saved = torch.load(out_ckpt, map_location=device, weights_only=False)
        best_epoch = saved.get("epoch", 0)
        best_dice = saved.get("val_macro_dice", 0.0)
        best_metrics = saved.get("best_metrics", {})
        duration = 0.0
        peak_vram = 0.0

    for epoch in range(1, epochs + 1):
        if skip_training:
            break
        model.train()
        train_loss = 0.0
        n_train = 0

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

        scheduler.step()

        # Validation
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

                m = compute_segmentation_metrics(preds, masks, num_classes=3)
                val_dice_sum += m["mean_dice"] * bs
                val_iou_sum += m["mean_iou"] * bs
                for k in metric_keys:
                    per_class_sums[k] += m.get(k, 0.0) * bs

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

        dis_d = epoch_metrics["dice_Diseased Tissue"]

        if epoch % 5 == 0 or epoch == 1 or val_dice > best_dice:
            mark = " *" if val_dice > best_dice else ""
            print(f"[{exp_name}] Ep {epoch:02d}: Loss={avg_train_loss:.4f}/{avg_val_loss:.4f} MacroDice={val_dice:.4f} DisDice={dis_d:.4f} MacroIoU={val_iou:.4f}{mark}")

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
                    "experiment": exp_name,
                    "class_weights": class_weights,
                    "loss_type": loss_type,
                    "image_size": image_size,
                    "status": "EXPERIMENTAL",
                },
                out_ckpt,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{exp_name}] Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    if not skip_training:
        duration = round(time.time() - t0, 1)
        peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if device.type == "cuda" else 0.0
    else:
        duration = 0.0
        peak_vram = 0.0

    with open(out_ckpt, "rb") as f:
        ckpt_sha = hashlib.sha256(f.read()).hexdigest()

    # Detailed evaluation on all 15 validation images at original 800x800 resolution
    eval_model = build_unet_resnet34(num_classes=3, encoder_name="resnet34", encoder_weights=None, checkpoint_path=out_ckpt, device=device)
    eval_model.eval()
    eval_tf = get_segmentation_eval_transforms(image_size)

    sample_evals = []
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    tot_healthy_fp = 0
    tot_healthy_leaf = 0

    for _, r in val_df.iterrows():
        fname = r["filename"]
        stem = Path(fname).stem
        cls_name = r["class_label"]
        img_bgr = cv2.imread(str(seg_img_dir / fname))
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(str(seg_mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

        inp = eval_tf(image=img_rgb)["image"].unsqueeze(0).to(device)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out_logits = eval_model(inp)
            pred_res = torch.argmax(out_logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        pred_orig = cv2.resize(pred_res, (w, h), interpolation=cv2.INTER_NEAREST)

        gt_leaf = (gt_mask >= 1)
        gt_dis = (gt_mask == 2)
        pr_leaf = (pred_orig >= 1)
        pr_dis = (pred_orig == 2)

        gt_leaf_px = int(gt_leaf.sum())
        gt_dis_px = int(gt_dis.sum())
        pr_leaf_px = int(pr_leaf.sum())
        pr_dis_px = int(pr_dis.sum())

        gt_pct = (gt_dis_px / max(1, gt_leaf_px)) * 100.0
        pr_pct = (pr_dis_px / max(1, pr_leaf_px)) * 100.0
        err_pct = abs(pr_pct - gt_pct)

        tp = int((gt_dis & pr_dis).sum())
        fp = int((~gt_dis & pr_dis).sum())
        fn = int((gt_dis & ~pr_dis).sum())

        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

        if cls_name == "Healthy Leaf":
            tot_healthy_fp += fp
            tot_healthy_leaf += gt_leaf_px

        dice_sample = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else (1.0 if gt_dis_px == 0 else 0.0)

        sample_evals.append({
            "filename": fname,
            "class_label": cls_name,
            "dice": round(dice_sample, 4),
            "gt_pct": round(gt_pct, 2),
            "pred_pct": round(pr_pct, 2),
            "area_err_pct": round(err_pct, 2),
            "fp_pixels": fp,
        })

    overall_prec = (tot_tp / max(1, tot_tp + tot_fp)) * 100.0
    overall_rec = (tot_tp / max(1, tot_tp + tot_fn)) * 100.0
    healthy_fp_rate = (tot_healthy_fp / max(1, tot_healthy_leaf)) * 100.0
    mean_area_err = np.mean([s["area_err_pct"] for s in sample_evals])

    res = {
        "experiment": exp_name,
        "image_size": image_size,
        "batch_size": batch_size,
        "loss_type": loss_type,
        "class_weights": list(class_weights),
        "best_epoch": best_epoch,
        "duration_sec": duration,
        "peak_vram_mb": peak_vram,
        "checkpoint_path": str(out_ckpt),
        "checkpoint_sha256": ckpt_sha,
        "val_macro_dice": best_metrics.get("val_macro_dice", 0.0),
        "val_macro_iou": best_metrics.get("val_macro_iou", 0.0),
        "val_background_dice": best_metrics.get("dice_Background", 0.0),
        "val_leaf_dice": best_metrics.get("dice_Healthy Leaf", 0.0),
        "val_disease_dice": best_metrics.get("dice_Diseased Tissue", 0.0),
        "val_disease_iou": best_metrics.get("iou_Diseased Tissue", 0.0),
        "disease_pixel_precision": round(overall_prec, 2),
        "disease_pixel_recall": round(overall_rec, 2),
        "healthy_fp_pixel_rate": round(healthy_fp_rate, 2),
        "mean_severity_error_pct": round(float(mean_area_err), 2),
        "sample_evals": sample_evals,
    }

    print(f"\n[{exp_name.upper()} RESULTS]")
    print(f"  Best Epoch:           {best_epoch}")
    print(f"  Macro Dice:           {res['val_macro_dice']:.4f}")
    print(f"  Disease Dice:         {res['val_disease_dice']:.4f}")
    print(f"  Disease IoU:          {res['val_disease_iou']:.4f}")
    print(f"  Disease Precision:    {res['disease_pixel_precision']:.2f}%")
    print(f"  Disease Recall:       {res['disease_pixel_recall']:.2f}%")
    print(f"  Healthy FP Area Rate: {res['healthy_fp_pixel_rate']:.2f}%")
    print(f"  Mean Severity Error:  {res['mean_severity_error_pct']:.2f}%")
    print(f"  Peak VRAM:            {peak_vram} MB")
    print(f"  Duration:             {duration} s")

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all", choices=["exp2", "exp3", "exp4a", "exp4b", "all"])
    parser.add_argument("--force", action="store_true", help="Force retrain even if checkpoint exists")
    args = parser.parse_args()

    results = {}
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_json = metrics_dir / "phase8_1_controlled_experiments.json"
    if out_json.is_file() and not args.force:
        try:
            with open(out_json, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    if args.exp in ("exp2", "all"):
        # Exp 2: 768x768, batch size 4 (to fit RTX 3050), Weighted CE + Dice (0.3, 1.0, 2.5)
        results["exp2_768"] = run_experiment("exp2_768", image_size=768, batch_size=4, loss_type="ce_dice", class_weights=(0.3, 1.0, 2.5), epochs=60, patience=15, force_retrain=args.force)

    if args.exp in ("exp3", "all"):
        # Exp 3: 512x512, batch size 8, Focal Loss + Dice (0.3, 1.0, 2.5)
        results["exp3_focal"] = run_experiment("exp3_focal", image_size=512, batch_size=8, loss_type="focal", class_weights=(0.3, 1.0, 2.5), epochs=60, patience=15, force_retrain=args.force)

    if args.exp in ("exp4a", "all"):
        # Exp 4a: 512x512, batch size 8, Weighted CE + Dice with disease weight 1.5
        results["exp4a_w15"] = run_experiment("exp4a_w15", image_size=512, batch_size=8, loss_type="ce_dice", class_weights=(0.3, 1.0, 1.5), epochs=60, patience=15, force_retrain=args.force)

    if args.exp in ("exp4b", "all"):
        # Exp 4b: 512x512, batch size 8, Weighted CE + Dice with disease weight 2.0
        results["exp4b_w20"] = run_experiment("exp4b_w20", image_size=512, batch_size=8, loss_type="ce_dice", class_weights=(0.3, 1.0, 2.0), epochs=60, patience=15, force_retrain=args.force)

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll experiments complete! Results saved to {out_json}")
