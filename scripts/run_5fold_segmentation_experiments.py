"""
CottonGuard AI — Strict 5-Fold Cross-Validation & Lesion Segmentation Improvement
Trains and evaluates U-Net ResNet-34 across all 5 folds defined in `proposed_70_image_stratified_folds.csv`.
Produces 100% out-of-fold predictions for all 70 approved development images.
Evaluates out-of-fold metrics, per-class breakdown, empty-GT behavior, and post-processing threshold ablations.
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

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet_resnet34 import (
    WeightedCombinedDiceCELoss,
    build_unet_resnet34,
    get_segmentation_eval_transforms,
    get_segmentation_train_transforms,
)


class AlbumentationsSegmentationDataset(Dataset):
    """Paired RGB Image + 3-Class Integer Mask dataset using Albumentations."""

    def __init__(self, records: List[dict], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        img_path = PROJECT_ROOT / rec["image_path"]
        mask_path = PROJECT_ROOT / rec["mask_path"]
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)
        augmented = self.transform(image=img, mask=mask)
        return augmented["image"].float(), augmented["mask"].long()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def post_process_mask_custom(
    raw_mask: np.ndarray,
    min_disease_cc_size: int = 50,
) -> np.ndarray:
    cleaned = raw_mask.copy()
    leaf_binary = (cleaned > 0).astype(np.uint8)
    leaf_filled = ndimage.binary_fill_holes(leaf_binary).astype(np.uint8)

    disease_binary = (cleaned == 2).astype(np.uint8)
    if min_disease_cc_size > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(disease_binary, connectivity=8)
        filtered_disease = np.zeros_like(disease_binary)
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area >= min_disease_cc_size:
                filtered_disease[labels == lbl] = 1
    else:
        filtered_disease = disease_binary

    # Strict leaf containment
    filtered_disease = (filtered_disease == 1) & (leaf_filled == 1)

    refined = np.zeros_like(raw_mask)
    refined[leaf_filled == 1] = 1
    refined[filtered_disease] = 2
    return refined


def compute_metrics_for_pair(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> Dict[str, Any]:
    # Leaf metrics (Classes 1 + 2)
    gt_leaf = (gt_mask > 0).astype(np.uint8)
    pred_leaf = (pred_mask > 0).astype(np.uint8)
    leaf_tp = int(np.sum((pred_leaf == 1) & (gt_leaf == 1)))
    leaf_fp = int(np.sum((pred_leaf == 1) & (gt_leaf == 0)))
    leaf_fn = int(np.sum((pred_leaf == 0) & (gt_leaf == 1)))
    leaf_union = leaf_tp + leaf_fp + leaf_fn
    leaf_iou = float(leaf_tp / leaf_union) if leaf_union > 0 else 1.0
    leaf_dice = float(2 * leaf_tp / (2 * leaf_tp + leaf_fp + leaf_fn)) if (2 * leaf_tp + leaf_fp + leaf_fn) > 0 else 1.0

    # Diseased Region metrics (Class 2)
    gt_dis = (gt_mask == 2).astype(np.uint8)
    pred_dis = (pred_mask == 2).astype(np.uint8)
    tp = int(np.sum((pred_dis == 1) & (gt_dis == 1)))
    fp = int(np.sum((pred_dis == 1) & (gt_dis == 0)))
    fn = int(np.sum((pred_dis == 0) & (gt_dis == 1)))
    gt_dis_px = int(np.sum(gt_dis))
    pred_dis_px = int(np.sum(pred_dis))

    # Explicit edge case handling
    # Case 1: GT empty & Pred empty -> perfect agreement on disease absence
    if gt_dis_px == 0 and pred_dis_px == 0:
        prec = 1.0
        rec = 1.0
        iou = 1.0
        dice = 1.0
    # Case 2: GT empty & Pred non-empty -> false alarms on healthy tissue
    elif gt_dis_px == 0 and pred_dis_px > 0:
        prec = 0.0
        rec = 0.0
        iou = 0.0
        dice = 0.0
    # Case 3: GT non-empty & Pred empty -> complete miss
    elif gt_dis_px > 0 and pred_dis_px == 0:
        prec = 0.0
        rec = 0.0
        iou = 0.0
        dice = 0.0
    # Standard non-empty overlap
    else:
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        union = tp + fp + fn
        iou = float(tp / union) if union > 0 else 0.0
        dice = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "leaf_dice": leaf_dice,
        "leaf_iou": leaf_iou,
        "disease_dice": dice,
        "disease_iou": iou,
        "precision": prec,
        "recall": rec,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_disease_pixels": gt_dis_px,
        "pred_disease_pixels": pred_dis_px,
        "gt_leaf_pixels": int(np.sum(gt_leaf)),
    }


def train_single_fold(
    fold_id: int,
    train_records: List[dict],
    val_records: List[dict],
    image_size: int = 512,
    batch_size: int = 8,
    epochs: int = 35,
    patience: int = 12,
    lr: float = 1e-4,
    device: torch.device = torch.device("cuda:0"),
    out_dir: Path = None,
) -> Tuple[Path, Dict[str, Any]]:
    set_seed(42 + fold_id)
    ckpt_path = out_dir / f"fold{fold_id}_best_unet.pt"

    train_ds = AlbumentationsSegmentationDataset(train_records, get_segmentation_train_transforms(image_size))
    val_ds = AlbumentationsSegmentationDataset(val_records, get_segmentation_eval_transforms(image_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = build_unet_resnet34(num_classes=3, encoder_name="resnet34", encoder_weights="imagenet", device=device)
    criterion = WeightedCombinedDiceCELoss(class_weights=(0.3, 1.0, 2.5), ce_weight=0.5, dice_weight=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_dice = -1.0
    best_epoch = 0
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_accum += float(loss.item())

        scheduler.step()
        train_loss = train_loss_accum / max(1, len(train_loader))

        # Evaluate on validation fold
        model.eval()
        val_dices = []
        with torch.inference_mode():
            for images, masks in val_loader:
                images = images.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
                gt = masks.numpy().astype(np.uint8)

                for b in range(preds.shape[0]):
                    p_dis = (preds[b] == 2)
                    g_dis = (gt[b] == 2)
                    inter = np.sum(p_dis & g_dis)
                    total = np.sum(p_dis) + np.sum(g_dis)
                    d = (2.0 * inter / total) if total > 0 else (1.0 if np.sum(g_dis) == 0 else 0.0)
                    val_dices.append(d)

        mean_val_dice = float(np.mean(val_dices))
        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "fold_id": fold_id,
                "model_state_dict": model.state_dict(),
                "val_dice": best_val_dice,
                "encoder": "resnet34",
                "classes": 3
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    duration = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

    return ckpt_path, {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_val_dice": round(best_val_dice, 4),
        "training_duration_s": round(duration, 2),
        "peak_vram_mb": round(peak_vram, 1),
        "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
    }


def main():
    print("=" * 80)
    print("COTTONGUARD AI - STRICT 5-FOLD CROSS-VALIDATION PIPELINE")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    manifest_p = PROJECT_ROOT / "artifacts" / "manifests" / "derived_70_image_dataset_manifest.json"
    folds_p = PROJECT_ROOT / "artifacts" / "manifests" / "proposed_70_image_stratified_folds.csv"

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    df_folds = pd.read_csv(folds_p)
    fold_map = dict(zip(df_folds["filename"], df_folds["fold_id"]))

    records = manifest["records"]
    for r in records:
        r["fold_id"] = fold_map[r["filename"]]

    print(f"Total development images: {len(records)}")
    folds_dir = PROJECT_ROOT / "artifacts" / "checkpoints" / "segmentation" / "5fold_cv"
    folds_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Train 5-fold cross-validation models
    fold_train_stats = []
    trained_models = {}

    for k in range(5):
        val_recs = [r for r in records if r["fold_id"] == k]
        train_recs = [r for r in records if r["fold_id"] != k]
        print(f"\n--- Training Fold {k} (Train: {len(train_recs)}, Val: {len(val_recs)}) ---")

        ckpt_path, stats = train_single_fold(
            fold_id=k,
            train_records=train_recs,
            val_records=val_recs,
            image_size=512,
            batch_size=8,
            epochs=35,
            patience=12,
            device=device,
            out_dir=folds_dir
        )
        fold_train_stats.append(stats)
        print(f"Fold {k} Best Epoch: {stats['best_epoch']} | Val Dice: {stats['best_val_dice']} | Duration: {stats['training_duration_s']}s | Peak VRAM: {stats['peak_vram_mb']} MB")

        # Load best model for inference
        m = build_unet_resnet34(num_classes=3, encoder_name="resnet34", encoder_weights=None, device=device)
        state = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(state["model_state_dict"])
        m.eval()
        trained_models[k] = m

    # Step 2: Full out-of-fold inference on all 70 images
    print("\n" + "=" * 80)
    print("RUNNING 100% OUT-OF-FOLD INFERENCE ACROSS ALL 70 DEVELOPMENT SAMPLES")
    print("=" * 80)

    raw_eval_transform = get_segmentation_eval_transforms(512)
    oof_predictions = []

    for r in records:
        k = r["fold_id"]
        fname = r["filename"]
        dclass = r["disease_class"]
        img_path = PROJECT_ROOT / r["image_path"]
        mask_path = PROJECT_ROOT / r["mask_path"]

        pil_img = Image.open(img_path).convert("RGB")
        img_rgb = np.array(pil_img)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = gt_mask.shape

        model = trained_models[k]
        transformed = raw_eval_transform(image=img_rgb)
        tensor = transformed["image"].unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_512 = np.argmax(probs, axis=0).astype(np.uint8)

        pred_orig = cv2.resize(pred_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        refined_pred = post_process_mask_custom(pred_orig, min_disease_cc_size=50)

        # Compute metrics
        m = compute_metrics_for_pair(gt_mask, refined_pred)
        m["filename"] = fname
        m["disease_class"] = dclass
        m["fold_id"] = k
        m["batch"] = r["batch"]
        oof_predictions.append(m)

    # Compute global aggregated metrics
    leaf_dices = [p["leaf_dice"] for p in oof_predictions]
    leaf_ious = [p["leaf_iou"] for p in oof_predictions]
    dis_dices = [p["disease_dice"] for p in oof_predictions]
    dis_ious = [p["disease_iou"] for p in oof_predictions]
    precisions = [p["precision"] for p in oof_predictions]
    recalls = [p["recall"] for p in oof_predictions]

    symptomatic_preds = [p for p in oof_predictions if p["gt_disease_pixels"] > 0]
    symp_dices = [p["disease_dice"] for p in symptomatic_preds]
    symp_ious = [p["disease_iou"] for p in symptomatic_preds]

    total_tp = sum(p["tp"] for p in oof_predictions)
    total_fp = sum(p["fp"] for p in oof_predictions)
    total_fn = sum(p["fn"] for p in oof_predictions)

    micro_prec = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    micro_dice = float(2 * total_tp / (2 * total_tp + total_fp + total_fn)) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0
    micro_iou = float(total_tp / (total_tp + total_fp + total_fn)) if (total_tp + total_fp + total_fn) > 0 else 0.0

    print("\n" + "=" * 80)
    print("STRICT 5-FOLD OUT-OF-FOLD GLOBAL METRICS (70 SAMPLES):")
    print(f"  Leaf-Level Macro Dice:             {np.mean(leaf_dices):.4f} ({np.mean(leaf_dices)*100:.2f}%)")
    print(f"  Leaf-Level Macro IoU:              {np.mean(leaf_ious):.4f} ({np.mean(leaf_ious)*100:.2f}%)")
    print(f"  Diseased-Region Macro Dice:        {np.mean(dis_dices):.4f} ({np.mean(dis_dices)*100:.2f}%)")
    print(f"  Diseased-Region Macro IoU:         {np.mean(dis_ious):.4f} ({np.mean(dis_ious)*100:.2f}%)")
    print(f"  Diseased-Region Macro Precision:   {np.mean(precisions):.4f} ({np.mean(precisions)*100:.2f}%)")
    print(f"  Diseased-Region Macro Recall:      {np.mean(recalls):.4f} ({np.mean(recalls)*100:.2f}%)")
    print(f"  Diseased-Region Micro Dice:        {micro_dice:.4f} ({micro_dice*100:.2f}%)")
    print(f"  Diseased-Region Micro IoU:         {micro_iou:.4f} ({micro_iou*100:.2f}%)")
    print(f"  Diseased-Region Micro Precision:   {micro_prec:.4f} ({micro_prec*100:.2f}%)")
    print(f"  Diseased-Region Micro Recall:      {micro_rec:.4f} ({micro_rec*100:.2f}%)")
    print(f"  Symptomatic Only Macro Dice:       {np.mean(symp_dices):.4f}")
    print(f"  Symptomatic Only Macro IoU:        {np.mean(symp_ious):.4f}")
    print("=" * 80)

    # Per-class breakdown
    print("\nPER-CLASS OUT-OF-FOLD PERFORMANCE:")
    print(f"{'Disease Class':<26} | {'N':<3} | {'Leaf Dice':<10} | {'Disease Dice':<12} | {'Disease IoU':<11} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 95)
    class_summary = {}
    for dclass in sorted(set(p["disease_class"] for p in oof_predictions)):
        sub = [p for p in oof_predictions if p["disease_class"] == dclass]
        c_leaf_dice = float(np.mean([p["leaf_dice"] for p in sub]))
        c_dis_dice = float(np.mean([p["disease_dice"] for p in sub]))
        c_dis_iou = float(np.mean([p["disease_iou"] for p in sub]))
        c_prec = float(np.mean([p["precision"] for p in sub]))
        c_rec = float(np.mean([p["recall"] for p in sub]))
        print(f"{dclass:<26} | {len(sub):<3} | {c_leaf_dice:<10.4f} | {c_dis_dice:<12.4f} | {c_dis_iou:<11.4f} | {c_prec:<10.4f} | {c_rec:<10.4f}")
        class_summary[dclass] = {
            "count": len(sub),
            "leaf_dice": round(c_leaf_dice, 4),
            "disease_dice": round(c_dis_dice, 4),
            "disease_iou": round(c_dis_iou, 4),
            "precision": round(c_prec, 4),
            "recall": round(c_rec, 4),
        }

    # Empty-GT behavior
    healthy_preds = [p for p in oof_predictions if p["disease_class"] == "Healthy Leaf"]
    perfect_zero_fp = sum(1 for p in healthy_preds if p["pred_disease_pixels"] == 0)
    mean_fp_px = float(np.mean([p["pred_disease_pixels"] for p in healthy_preds]))
    tot_healthy_fp_px = sum(p["pred_disease_pixels"] for p in healthy_preds)
    tot_healthy_leaf_px = sum(p["gt_leaf_pixels"] for p in healthy_preds)
    agg_fp_ratio = tot_healthy_fp_px / tot_healthy_leaf_px if tot_healthy_leaf_px > 0 else 0.0

    print(f"\nEMPTY-GT BEHAVIOR (Healthy Leaf Control):")
    print(f"  Total Healthy Images:              {len(healthy_preds)}")
    print(f"  Zero FP Samples:                   {perfect_zero_fp} / {len(healthy_preds)}")
    print(f"  Mean FP Disease Pixels:            {mean_fp_px:.1f} px")
    print(f"  Aggregate FP / Leaf Ratio:         {agg_fp_ratio:.4f} ({agg_fp_ratio*100:.2f}%)")

    # Save comprehensive metrics JSON
    out_json = {
        "timestamp": datetime.now().isoformat(),
        "methodology": "5-Fold Stratified Cross-Validation (100% Out-Of-Fold)",
        "model_architecture": "U-Net with ResNet-34 Encoder",
        "resolution": 512,
        "total_samples": len(oof_predictions),
        "fold_training_statistics": fold_train_stats,
        "global_metrics": {
            "leaf_macro_dice": round(float(np.mean(leaf_dices)), 4),
            "leaf_macro_iou": round(float(np.mean(leaf_ious)), 4),
            "disease_macro_dice": round(float(np.mean(dis_dices)), 4),
            "disease_macro_iou": round(float(np.mean(dis_ious)), 4),
            "disease_macro_precision": round(float(np.mean(precisions)), 4),
            "disease_macro_recall": round(float(np.mean(recalls)), 4),
            "disease_micro_dice": round(micro_dice, 4),
            "disease_micro_iou": round(micro_iou, 4),
            "disease_micro_precision": round(micro_prec, 4),
            "disease_micro_recall": round(micro_rec, 4),
            "symptomatic_macro_dice": round(float(np.mean(symp_dices)), 4),
            "symptomatic_macro_iou": round(float(np.mean(symp_ious)), 4),
            "total_tp_pixels": total_tp,
            "total_fp_pixels": total_fp,
            "total_fn_pixels": total_fn
        },
        "per_class_summary": class_summary,
        "empty_gt_behavior": {
            "healthy_leaf_samples": len(healthy_preds),
            "zero_false_positive_samples": perfect_zero_fp,
            "mean_false_positive_pixels": round(mean_fp_px, 1),
            "aggregate_false_positive_ratio": round(agg_fp_ratio, 4)
        },
        "per_sample_oof_results": oof_predictions
    }

    metrics_out = PROJECT_ROOT / "artifacts" / "metrics" / "five_fold_cross_validation_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[DONE] Saved 5-fold cross-validation metrics to: {metrics_out}")

    # Step 3: Post-processing threshold ablation
    print("\n" + "=" * 80)
    print("RUNNING POST-PROCESSING CONNECTED-COMPONENT THRESHOLD ABLATION")
    print("=" * 80)

    threshold_results = []
    cc_candidates = [0, 15, 30, 50, 75]

    for cc in cc_candidates:
        cc_dices = []
        bb_dices = []
        healthy_fps = []
        for r in records:
            k = r["fold_id"]
            img_path = PROJECT_ROOT / r["image_path"]
            mask_path = PROJECT_ROOT / r["mask_path"]
            pil_img = Image.open(img_path).convert("RGB")
            img_rgb = np.array(pil_img)
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            orig_h, orig_w = gt_mask.shape

            model = trained_models[k]
            transformed = raw_eval_transform(image=img_rgb)
            tensor = transformed["image"].unsqueeze(0).to(device)

            with torch.inference_mode():
                logits = model(tensor)
                pred_512 = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

            pred_orig = cv2.resize(pred_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            refined = post_process_mask_custom(pred_orig, min_disease_cc_size=cc)
            res = compute_metrics_for_pair(gt_mask, refined)

            cc_dices.append(res["disease_dice"])
            if r["disease_class"] == "Bacterial Blight":
                bb_dices.append(res["disease_dice"])
            if r["disease_class"] == "Healthy Leaf":
                healthy_fps.append(res["pred_disease_pixels"])

        mean_dice = float(np.mean(cc_dices))
        mean_bb_dice = float(np.mean(bb_dices))
        mean_h_fp = float(np.mean(healthy_fps))
        print(f"CC Threshold: {cc:>2} px | Overall Macro Dice: {mean_dice:.4f} | Bacterial Blight Dice: {mean_bb_dice:.4f} | Healthy Mean FP: {mean_h_fp:.1f} px")
        threshold_results.append({
            "min_cc_size": cc,
            "overall_macro_dice": round(mean_dice, 4),
            "bacterial_blight_dice": round(mean_bb_dice, 4),
            "healthy_leaf_mean_fp_pixels": round(mean_h_fp, 1),
        })

    ablation_out = PROJECT_ROOT / "artifacts" / "metrics" / "post_processing_threshold_ablation.json"
    with open(ablation_out, "w", encoding="utf-8") as f:
        json.dump(threshold_results, f, indent=2)
    print(f"[DONE] Saved ablation results to: {ablation_out}")


if __name__ == "__main__":
    main()
