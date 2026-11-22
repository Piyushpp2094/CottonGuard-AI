"""
CottonGuard AI — Diseased-Region Segmentation Detailed Evaluation
Computes Dice, IoU, Precision, Recall, and confusion metrics for Class 2 (Diseased Region)
across all 70 development images, broken down by disease class, empty-GT behavior, and pixel counts.
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet_resnet34 import UNetResNet34Segmentor, get_segmentation_eval_transforms


def post_process_mask(raw_mask: np.ndarray, min_disease_cc_size: int = 50) -> np.ndarray:
    cleaned = raw_mask.copy()
    leaf_binary = (cleaned > 0).astype(np.uint8)
    leaf_filled = ndimage.binary_fill_holes(leaf_binary).astype(np.uint8)

    disease_binary = (cleaned == 2).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(disease_binary, connectivity=8)

    filtered_disease = np.zeros_like(disease_binary)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_disease_cc_size:
            filtered_disease[labels == lbl] = 1

    # Strict leaf containment
    filtered_disease = (filtered_disease == 1) & (leaf_filled == 1)

    refined = np.zeros_like(raw_mask)
    refined[leaf_filled == 1] = 1
    refined[filtered_disease] = 2
    return refined


def main():
    print("=" * 80)
    print("COTTONGUARD AI - DETAILED DISEASED-REGION SEGMENTATION AUDIT")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    manifest_path = PROJECT_ROOT / "artifacts" / "manifests" / "derived_70_image_dataset_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = manifest["records"]
    print(f"Total development images in derived manifest: {len(records)}")

    segmentor = UNetResNet34Segmentor(image_size=512, device=device)
    raw_eval_transform = get_segmentation_eval_transforms(512)
    model = segmentor._ensure_loaded()
    model.eval()

    per_image_results = []
    class_aggregates = defaultdict(lambda: {
        "count": 0,
        "total_gt_disease_px": 0,
        "total_pred_disease_px": 0,
        "tp": 0, "fp": 0, "fn": 0,
        "dices": [], "ious": [], "precisions": [], "recalls": []
    })

    total_tp, total_fp, total_fn = 0, 0, 0
    all_dices, all_ious, all_precisions, all_recalls = [], [], [], []
    symptomatic_dices, symptomatic_ious = [], []

    for idx, rec in enumerate(records, 1):
        fname = rec["filename"]
        dclass = rec["disease_class"]
        img_path = PROJECT_ROOT / rec["image_path"]
        gt_mask_path = PROJECT_ROOT / rec["mask_path"]

        with Image.open(img_path) as pil_img:
            img_rgb = np.array(pil_img.convert("RGB"))

        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = gt_mask.shape

        # Model inference at 512x512
        transformed = raw_eval_transform(image=img_rgb)
        tensor = transformed["image"].unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(tensor)
            pred_512 = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        # Nearest neighbor resize to original resolution
        pred_orig = cv2.resize(pred_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        refined_pred = post_process_mask(pred_orig, min_disease_cc_size=50)

        # Extract Class 2 (Diseased Region)
        gt_disease = (gt_mask == 2).astype(np.uint8)
        pred_disease = (refined_pred == 2).astype(np.uint8)

        tp = int(np.sum((pred_disease == 1) & (gt_disease == 1)))
        fp = int(np.sum((pred_disease == 1) & (gt_disease == 0)))
        fn = int(np.sum((pred_disease == 0) & (gt_disease == 1)))
        gt_px = int(np.sum(gt_disease == 1))
        pred_px = int(np.sum(pred_disease == 1))

        # Precision & Recall
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if gt_px == 0 else 0.0)
        rec_val = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if pred_px == 0 else 0.0)

        # Dice & IoU
        intersection = tp
        union = tp + fp + fn
        iou = float(intersection / union) if union > 0 else (1.0 if (gt_px == 0 and pred_px == 0) else 0.0)
        dice = float(2 * intersection / (2 * intersection + fp + fn)) if (2 * intersection + fp + fn) > 0 else (1.0 if (gt_px == 0 and pred_px == 0) else 0.0)

        # Record
        total_tp += tp
        total_fp += fp
        total_fn += fn

        all_dices.append(dice)
        all_ious.append(iou)
        all_precisions.append(prec)
        all_recalls.append(rec_val)

        if gt_px > 0:
            symptomatic_dices.append(dice)
            symptomatic_ious.append(iou)

        # Group stats
        grp = class_aggregates[dclass]
        grp["count"] += 1
        grp["total_gt_disease_px"] += gt_px
        grp["total_pred_disease_px"] += pred_px
        grp["tp"] += tp
        grp["fp"] += fp
        grp["fn"] += fn
        grp["dices"].append(dice)
        grp["ious"].append(iou)
        grp["precisions"].append(prec)
        grp["recalls"].append(rec_val)

        per_image_results.append({
            "filename": fname,
            "disease_class": dclass,
            "gt_disease_pixels": gt_px,
            "pred_disease_pixels": pred_px,
            "tp": tp, "fp": fp, "fn": fn,
            "dice": round(dice, 4),
            "iou": round(iou, 4),
            "precision": round(prec, 4),
            "recall": round(rec_val, 4)
        })

    # Summary metrics
    # Micro (aggregated across all pixels)
    micro_prec = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    micro_iou = float(total_tp / (total_tp + total_fp + total_fn)) if (total_tp + total_fp + total_fn) > 0 else 0.0
    micro_dice = float(2 * total_tp / (2 * total_tp + total_fp + total_fn)) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0

    # Macro (average across samples)
    macro_dice = float(np.mean(all_dices))
    macro_iou = float(np.mean(all_ious))
    macro_prec = float(np.mean(all_precisions))
    macro_rec = float(np.mean(all_recalls))

    symptomatic_macro_dice = float(np.mean(symptomatic_dices)) if symptomatic_dices else 0.0
    symptomatic_macro_iou = float(np.mean(symptomatic_ious)) if symptomatic_ious else 0.0

    print("\n" + "=" * 80)
    print("GLOBAL DISEASED-REGION SEGMENTATION METRICS (70 Development Samples):")
    print(f"  Macro Diseased-Region Dice:        {macro_dice:.4f} ({macro_dice * 100:.2f}%)")
    print(f"  Macro Diseased-Region IoU:         {macro_iou:.4f} ({macro_iou * 100:.2f}%)")
    print(f"  Macro Precision:                   {macro_prec:.4f} ({macro_prec * 100:.2f}%)")
    print(f"  Macro Recall:                      {macro_rec:.4f} ({macro_rec * 100:.2f}%)")
    print(f"  Micro Pixel-Aggregated Dice:       {micro_dice:.4f} ({micro_dice * 100:.2f}%)")
    print(f"  Micro Pixel-Aggregated IoU:        {micro_iou:.4f} ({micro_iou * 100:.2f}%)")
    print(f"  Micro Precision:                   {micro_prec:.4f} ({micro_prec * 100:.2f}%)")
    print(f"  Micro Recall:                      {micro_rec:.4f} ({micro_rec * 100:.2f}%)")
    print(f"  Symptomatic Only Macro Dice:       {symptomatic_macro_dice:.4f}")
    print(f"  Symptomatic Only Macro IoU:        {symptomatic_macro_iou:.4f}")
    print("=" * 80)

    # Class breakdown
    class_summary = {}
    print("\nPER-CLASS DISEASED-REGION BREAKDOWN:")
    print(f"{'Disease Class':<26} | {'N':<3} | {'Mean Dice':<10} | {'Mean IoU':<10} | {'Precision':<10} | {'Recall':<10} | {'GT Disease Px':<13} | {'Pred Px':<10}")
    print("-" * 105)
    for cname, stats in sorted(class_aggregates.items()):
        m_dice = float(np.mean(stats["dices"]))
        m_iou = float(np.mean(stats["ious"]))
        m_prec = float(np.mean(stats["precisions"]))
        m_rec = float(np.mean(stats["recalls"]))
        print(f"{cname:<26} | {stats['count']:<3} | {m_dice:<10.4f} | {m_iou:<10.4f} | {m_prec:<10.4f} | {m_rec:<10.4f} | {stats['total_gt_disease_px']:<13,d} | {stats['total_pred_disease_px']:<10,d}")
        class_summary[cname] = {
            "count": stats["count"],
            "mean_dice": round(m_dice, 4),
            "mean_iou": round(m_iou, 4),
            "mean_precision": round(m_prec, 4),
            "mean_recall": round(m_rec, 4),
            "total_gt_disease_px": stats["total_gt_disease_px"],
            "total_pred_disease_px": stats["total_pred_disease_px"]
        }

    # Empty GT audit (Healthy leaves where GT disease is 0)
    healthy_audit = [res for res in per_image_results if res["disease_class"] == "Healthy Leaf"]
    empty_gt_perfect = sum(1 for res in healthy_audit if res["pred_disease_pixels"] == 0)
    print(f"\nEMPTY-GT BEHAVIOR (Healthy Leaf Control):")
    print(f"  Total Healthy Images:              {len(healthy_audit)}")
    print(f"  Images with exactly 0 pred disease:{empty_gt_perfect} / {len(healthy_audit)}")
    print(f"  Mean FP disease pixels on healthy: {np.mean([r['pred_disease_pixels'] for r in healthy_audit]):.1f} px")

    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": "artifacts/checkpoints/segmentation/best_unet.pt",
        "architecture": "U-Net with ResNet-34 Encoder",
        "total_samples": len(records),
        "global_metrics": {
            "macro_dice": round(macro_dice, 4),
            "macro_iou": round(macro_iou, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "micro_dice": round(micro_dice, 4),
            "micro_iou": round(micro_iou, 4),
            "micro_precision": round(micro_prec, 4),
            "micro_recall": round(micro_rec, 4),
            "symptomatic_only_macro_dice": round(symptomatic_macro_dice, 4),
            "symptomatic_only_macro_iou": round(symptomatic_macro_iou, 4),
            "total_tp_pixels": total_tp,
            "total_fp_pixels": total_fp,
            "total_fn_pixels": total_fn
        },
        "per_class_summary": class_summary,
        "empty_gt_behavior": {
            "healthy_leaf_samples": len(healthy_audit),
            "zero_false_positive_samples": empty_gt_perfect,
            "mean_false_positive_pixels": round(float(np.mean([r["pred_disease_pixels"] for r in healthy_audit])), 1)
        },
        "per_image_results": per_image_results
    }

    out_file = PROJECT_ROOT / "artifacts" / "metrics" / "diseased_region_segmentation_metrics.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[DONE] Saved comprehensive diseased-region metrics to: {out_file}")


if __name__ == "__main__":
    main()
