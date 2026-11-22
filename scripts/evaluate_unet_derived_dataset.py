"""
CottonGuard AI — Phase 3 U-Net Segmentation Evaluation & Post-Processing Pipeline
Evaluates best_unet.pt against the 70 development derived ground-truth masks.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet_resnet34 import UNetResNet34Segmentor, build_unet_resnet34, get_segmentation_eval_transforms


def post_process_mask(raw_mask: np.ndarray, min_disease_cc_size: int = 50) -> np.ndarray:
    """
    Morphological refinement:
    1. Leaf mask extraction (classes 1 and 2).
    2. Hole filling on leaf mask to close internal gaps.
    3. Remove small spurious connected components (< 50 px) in diseased mask (class 2).
    4. Enforce strict containment: Diseased region must reside strictly within leaf.
    """
    cleaned = raw_mask.copy()
    leaf_binary = (cleaned > 0).astype(np.uint8)
    
    # 1. Leaf hole filling
    leaf_filled = ndimage.binary_fill_holes(leaf_binary).astype(np.uint8)
    
    # 2. Diseased region connected components filter
    disease_binary = (cleaned == 2).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(disease_binary, connectivity=8)
    
    filtered_disease = np.zeros_like(disease_binary)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_disease_cc_size:
            filtered_disease[labels == lbl] = 1

    # 3. Containment: Diseased must be inside leaf_filled
    filtered_disease = (filtered_disease == 1) & (leaf_filled == 1)
    
    # 4. Reconstruct refined 3-class mask
    refined = np.zeros_like(raw_mask)
    refined[leaf_filled == 1] = 1
    refined[filtered_disease] = 2
    
    return refined


def compute_binary_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Dict[str, float]:
    pred_pos = (pred_bin == 1)
    gt_pos = (gt_bin == 1)
    
    tp = float(np.sum(pred_pos & gt_pos))
    fp = float(np.sum(pred_pos & ~gt_pos))
    fn = float(np.sum(~pred_pos & gt_pos))
    tn = float(np.sum(~pred_pos & ~gt_pos))
    
    intersection = tp
    union = tp + fp + fn
    
    iou = float(intersection / union) if union > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)
    dice = float((2.0 * intersection) / (2.0 * intersection + fp + fn)) if (2.0 * intersection + fp + fn) > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn)
    }


def evaluate_segmentation():
    print("=" * 70)
    print("COTTON GUARD AI — PHASE 3 U-NET SEGMENTATION EVALUATION")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    manifest_path = PROJECT_ROOT / "artifacts" / "manifests" / "derived_70_image_dataset_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    records = manifest["records"]
    print(f"Evaluating {len(records)} development image-mask pairs...")

    segmentor = UNetResNet34Segmentor(image_size=512, device=device)
    raw_eval_transform = get_segmentation_eval_transforms(512)
    model = segmentor._ensure_loaded()

    # Create visual review output dir
    visuals_dir = PROJECT_ROOT / "artifacts" / "metrics" / "segmentation_eval_visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    per_sample_metrics = []
    
    # Aggregated confusion elements
    leaf_dices, leaf_ious = [], []
    disease_dices, disease_ious = [], []
    healthy_leaf_false_disease_px = []
    
    gpu_latencies = []

    for idx, rec in enumerate(records):
        fname = rec["filename"]
        dclass = rec["disease_class"]
        img_rel = rec["image_path"]
        mask_rel = rec["mask_path"]

        img_path = PROJECT_ROOT / img_rel
        gt_mask_path = PROJECT_ROOT / mask_rel

        with Image.open(img_path) as pil_img:
            img_rgb = np.array(pil_img.convert("RGB"))
        
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = gt_mask.shape

        # Resize to 512x512 for U-Net evaluation
        transformed = raw_eval_transform(image=img_rgb)
        tensor = transformed["image"].unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(tensor)
            pred_512 = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        gpu_latencies.append((t1 - t0) * 1000.0)

        # Resize predicted mask back to original image resolution with Nearest Neighbor
        pred_orig_res = cv2.resize(pred_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # Apply morphological post-processing
        refined_pred = post_process_mask(pred_orig_res, min_disease_cc_size=50)

        # Compute metrics for Leaf (Class 1 or 2)
        gt_leaf = (gt_mask > 0).astype(np.uint8)
        pred_leaf = (refined_pred > 0).astype(np.uint8)
        leaf_res = compute_binary_metrics(pred_leaf, gt_leaf)
        leaf_dices.append(leaf_res["dice"])
        leaf_ious.append(leaf_res["iou"])

        # Compute metrics for Diseased Region (Class 2)
        gt_disease = (gt_mask == 2).astype(np.uint8)
        pred_disease = (refined_pred == 2).astype(np.uint8)
        disease_res = compute_binary_metrics(pred_disease, gt_disease)
        
        # Only include disease metrics in average if disease is present, or count 1.0 for true negative
        disease_dices.append(disease_res["dice"])
        disease_ious.append(disease_res["iou"])

        # Check Healthy Leaf false positive behavior
        if dclass == "Healthy Leaf":
            fp_px = int(np.sum(pred_disease == 1))
            total_leaf = int(np.sum(pred_leaf == 1))
            fp_ratio = float(fp_px / max(1, total_leaf))
            healthy_leaf_false_disease_px.append({
                "filename": fname,
                "false_positive_disease_pixels": fp_px,
                "total_leaf_pixels": total_leaf,
                "false_positive_ratio": round(fp_ratio, 4)
            })

        # Save visual comparison for sample frames
        if idx in [0, 5, 12, 15, 21, 22, 24, 31, 50, 60]:
            # Create comparison: GT vs Pred
            vis = np.zeros((orig_h, orig_w * 2 + 10, 3), dtype=np.uint8)
            # GT colorized
            gt_color = img_rgb.copy()
            gt_color[gt_mask == 1] = (gt_color[gt_mask == 1] * 0.5 + np.array([34, 197, 94]) * 0.5).astype(np.uint8)
            gt_color[gt_mask == 2] = (gt_color[gt_mask == 2] * 0.5 + np.array([239, 68, 68]) * 0.5).astype(np.uint8)
            
            # Pred colorized
            pred_color = img_rgb.copy()
            pred_color[refined_pred == 1] = (pred_color[refined_pred == 1] * 0.5 + np.array([34, 197, 94]) * 0.5).astype(np.uint8)
            pred_color[refined_pred == 2] = (pred_color[refined_pred == 2] * 0.5 + np.array([239, 68, 68]) * 0.5).astype(np.uint8)

            vis[:, :orig_w] = gt_color
            vis[:, orig_w:orig_w+10] = 50
            vis[:, orig_w+10:] = pred_color
            cv2.imwrite(str(visuals_dir / f"eval_vis_{idx:02d}_{Path(fname).stem}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        per_sample_metrics.append({
            "global_id": rec["global_id"],
            "filename": fname,
            "disease_class": dclass,
            "leaf_dice": round(leaf_res["dice"], 4),
            "leaf_iou": round(leaf_res["iou"], 4),
            "disease_dice": round(disease_res["dice"], 4),
            "disease_iou": round(disease_res["iou"], 4),
        })

    # Benchmark CPU latency
    print("Benchmarking CPU latency...")
    model_cpu = segmentor.device = torch.device("cpu")
    model_c = build_unet_resnet34(num_classes=3, encoder_name="resnet34", encoder_weights=None, checkpoint_path=segmentor.checkpoint_path, device=torch.device("cpu"))
    model_c.eval()
    cpu_dummy = torch.randn(1, 3, 512, 512)
    cpu_times = []
    with torch.no_grad():
        for _ in range(5):
            t0 = time.perf_counter()
            _ = model_c(cpu_dummy)
            t1 = time.perf_counter()
            cpu_times.append((t1 - t0) * 1000.0)
    cpu_latency_mean = sum(cpu_times) / len(cpu_times)

    # VRAM footprint
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        dummy_gpu = torch.randn(1, 3, 512, 512, device=device)
        _ = model(dummy_gpu)
        if device.type == "cuda":
            torch.cuda.synchronize()
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    mean_leaf_dice = float(np.mean(leaf_dices))
    mean_leaf_iou = float(np.mean(leaf_ious))
    mean_disease_dice = float(np.mean(disease_dices))
    mean_disease_iou = float(np.mean(disease_ious))
    overall_mean_dice = (mean_leaf_dice + mean_disease_dice) / 2.0
    overall_mean_iou = (mean_leaf_iou + mean_disease_iou) / 2.0

    eval_results = {
        "checkpoint": "artifacts/checkpoints/segmentation/best_unet.pt",
        "architecture": "UNet-ResNet34 (smp)",
        "input_resolution": [512, 512],
        "samples_evaluated": len(records),
        "overall_metrics": {
            "mean_dice_overall": round(overall_mean_dice, 4),
            "mean_iou_overall": round(overall_mean_iou, 4),
            "leaf_dice": round(mean_leaf_dice, 4),
            "leaf_iou": round(mean_leaf_iou, 4),
            "diseased_region_dice": round(mean_disease_dice, 4),
            "diseased_region_iou": round(mean_disease_iou, 4)
        },
        "healthy_leaf_control_audit": {
            "total_healthy_images": len(healthy_leaf_false_disease_px),
            "samples": healthy_leaf_false_disease_px,
            "mean_false_positive_ratio": round(float(np.mean([s['false_positive_ratio'] for s in healthy_leaf_false_disease_px])), 4) if healthy_leaf_false_disease_px else 0.0
        },
        "hardware_telemetry": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "None",
            "mean_gpu_latency_ms": round(float(np.mean(gpu_latencies)), 2),
            "p95_gpu_latency_ms": round(float(np.percentile(gpu_latencies, 95)), 2),
            "mean_cpu_latency_ms": round(cpu_latency_mean, 2),
            "peak_vram_mb": round(peak_vram_mb, 2)
        },
        "per_sample_summary": per_sample_metrics
    }

    out_json = PROJECT_ROOT / "artifacts" / "metrics" / "segmentation_evaluation.json"
    with open(out_json, "w") as f:
        json.dump(eval_results, f, indent=2)

    print("\n" + "=" * 50)
    print("SEGMENTATION EVALUATION RESULTS:")
    print(f"  Overall Mean Dice:        {overall_mean_dice:.4f}")
    print(f"  Overall Mean IoU:         {overall_mean_iou:.4f}")
    print(f"  Leaf Class Dice:          {mean_leaf_dice:.4f}")
    print(f"  Leaf Class IoU:           {mean_leaf_iou:.4f}")
    print(f"  Diseased Region Dice:     {mean_disease_dice:.4f}")
    print(f"  Diseased Region IoU:      {mean_disease_iou:.4f}")
    print(f"  Healthy Leaf FP Ratio:    {eval_results['healthy_leaf_control_audit']['mean_false_positive_ratio']:.4f}")
    print(f"  GPU Inference Latency:    {eval_results['hardware_telemetry']['mean_gpu_latency_ms']} ms")
    print(f"  CPU Inference Latency:    {eval_results['hardware_telemetry']['mean_cpu_latency_ms']} ms")
    print(f"  Peak VRAM Footprint:      {eval_results['hardware_telemetry']['peak_vram_mb']} MB")
    print("=" * 50)
    print(f"[SUCCESS] Metrics saved to {out_json}")


if __name__ == "__main__":
    evaluate_segmentation()
