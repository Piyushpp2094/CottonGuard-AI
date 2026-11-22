"""
CottonGuard AI — Phase 5 Model Explainability & Multi-Layer Visual Stack Generator
Generates full 5-layer diagnostic visual stack across all 7 disease classes:
1. Layer 1: Clean Original Image
2. Layer 2: Grad-CAM Saliency Heatmap (Turbo/Jet colormap blend)
3. Layer 3: Leaf Boundary Contour Mask (Isolated foliar boundary)
4. Layer 4: Lesion Segmentation Mask (Class 2 diseased tissue)
5. Layer 5: Comprehensive Diagnostic Composite (Multi-panel presentation standard)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import build_classifier
from src.models.unet_resnet34 import UNetResNet34Segmentor
from src.explainability.gradcam import GradCAM
from src.explainability.visualization import generate_heatmap_image, generate_overlay_image


def generate_showcase():
    print("=" * 70)
    print("COTTON GUARD AI — PHASE 5 MULTI-LAYER EXPLAINABILITY SHOWCASE")
    print("=" * 70)

    device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    print(f"GPU Device: {device_gpu} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'})")

    out_dir = PROJECT_ROOT / "artifacts" / "explainability_showcase"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Classifier & Grad-CAM
    ckpt_cls = PROJECT_ROOT / "artifacts" / "checkpoints" / "classification" / "best_model.pt"
    classifier = build_classifier(model_name="efficientnet_b0", num_classes=7, checkpoint_path=ckpt_cls, device=device_gpu)
    classifier.eval()
    gradcam_gpu = GradCAM(model=classifier, device=device_gpu)

    # CPU Explainer for benchmark
    classifier_cpu = build_classifier(model_name="efficientnet_b0", num_classes=7, checkpoint_path=ckpt_cls, device=device_cpu)
    classifier_cpu.eval()
    gradcam_cpu = GradCAM(model=classifier_cpu, device=device_cpu)

    # 2. Load Segmentor
    segmentor = UNetResNet34Segmentor(image_size=512, device=device_gpu)
    _ = segmentor._ensure_loaded()

    # 3. Select 7 representative images from derived dataset manifest (one per class)
    manifest_path = PROJECT_ROOT / "artifacts" / "manifests" / "derived_70_image_dataset_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    records = manifest["records"]

    # Pick representative image for each class
    selected_by_class = {}
    for r in records:
        dclass = r["disease_class"]
        if dclass not in selected_by_class:
            selected_by_class[dclass] = r

    classes = [
        "Bacterial Blight",
        "Curl Virus",
        "Healthy Leaf",
        "Herbicide Growth Damage",
        "Leaf Hopper Jassids",
        "Leaf Reddening",
        "Leaf Variegation"
    ]

    gpu_latencies = []
    cpu_latencies = []
    results_summary = []

    for c_idx, cls_name in enumerate(classes):
        rec = selected_by_class.get(cls_name)
        if not rec:
            continue

        fname = rec["filename"]
        stem = Path(fname).stem
        img_path = PROJECT_ROOT / rec["image_path"]

        with Image.open(img_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            w, h = rgb_img.size
            img_np = np.array(rgb_img)

        # Benchmark GPU Grad-CAM latency
        lat_gpu_runs = []
        for _ in range(5):
            if device_gpu.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            cam_res = gradcam_gpu.generate_cam(rgb_img)
            if device_gpu.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            lat_gpu_runs.append((t1 - t0) * 1000.0)
        mean_gpu_lat = sum(lat_gpu_runs) / len(lat_gpu_runs)
        gpu_latencies.append(mean_gpu_lat)

        # Benchmark CPU Grad-CAM latency
        t0 = time.perf_counter()
        _ = gradcam_cpu.generate_cam(rgb_img)
        t1 = time.perf_counter()
        mean_cpu_lat = (t1 - t0) * 1000.0
        cpu_latencies.append(mean_cpu_lat)

        heatmap = cam_res["heatmap"]
        assert not np.isnan(heatmap).any(), f"NaN detected in heatmap for {fname}"
        assert not np.isinf(heatmap).any(), f"Inf detected in heatmap for {fname}"

        pred_class = cam_res["predicted_class_name"]
        conf = cam_res["confidence"]

        # Run Segmentation
        seg_res = segmentor.predict(rgb_img)
        seg_mask = seg_res.get("mask")
        if seg_mask is None or seg_mask.shape != (h, w):
            seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST) if seg_mask is not None else np.zeros((h, w), dtype=np.uint8)

        # ---- GENERATE 5 VISUAL PRESENTATION LAYERS ----
        
        # Layer 1: Clean Original Image
        layer1_path = out_dir / f"{c_idx+1:02d}_{stem}_layer1_original.jpg"
        rgb_img.save(layer1_path, quality=95)

        # Layer 2: Grad-CAM Saliency Overlay (Turbo colormap)
        overlay_cam = generate_overlay_image(rgb_img, heatmap, alpha=0.45, colormap="turbo")
        layer2_path = out_dir / f"{c_idx+1:02d}_{stem}_layer2_gradcam.jpg"
        overlay_cam.save(layer2_path, quality=95)

        # Layer 3: Leaf Boundary / Contour Mask
        leaf_mask_binary = (seg_mask > 0).astype(np.uint8)
        layer3_img = img_np.copy()
        contours_leaf, _ = cv2.findContours(leaf_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(layer3_img, contours_leaf, -1, (34, 197, 94), 3) # Emerald green contour
        layer3_path = out_dir / f"{c_idx+1:02d}_{stem}_layer3_leaf_boundary.jpg"
        Image.fromarray(layer3_img).save(layer3_path, quality=95)

        # Layer 4: Lesion Segmentation Mask
        disease_mask_binary = (seg_mask == 2).astype(np.uint8)
        layer4_img = np.zeros_like(img_np)
        layer4_img[disease_mask_binary == 1] = [239, 68, 68] # Ruby red
        # Add subtle original background
        layer4_blend = (img_np * 0.35 + layer4_img * 0.65).astype(np.uint8)
        layer4_blend[disease_mask_binary == 0] = (img_np[disease_mask_binary == 0] * 0.25).astype(np.uint8)
        layer4_path = out_dir / f"{c_idx+1:02d}_{stem}_layer4_lesion_mask.jpg"
        Image.fromarray(layer4_blend).save(layer4_path, quality=95)

        # Layer 5: Diagnostic Composite Presentation Panel
        # Create a 4-panel side-by-side composite: [Original | Grad-CAM Saliency | Leaf Boundary | Lesion Contours]
        panel_w = 400
        panel_h = int(h * (panel_w / w))
        
        p1 = cv2.resize(img_np, (panel_w, panel_h))
        p2 = cv2.resize(np.array(overlay_cam), (panel_w, panel_h))
        p3 = cv2.resize(layer3_img, (panel_w, panel_h))
        p4 = cv2.resize(layer4_blend, (panel_w, panel_h))

        # Add headers to each panel
        def add_header(img_arr, title):
            hdr = np.zeros((40, panel_w, 3), dtype=np.uint8) + 30
            cv2.putText(hdr, title, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            return np.vstack([hdr, img_arr])

        c1 = add_header(p1, "1. Input Photograph")
        c2 = add_header(p2, "2. Grad-CAM Saliency")
        c3 = add_header(p3, "3. Leaf Boundary")
        c4 = add_header(p4, "4. Lesion Contours")

        composite_grid = np.hstack([c1, c2, c3, c4])
        
        # Add master banner on top
        banner = np.zeros((65, composite_grid.shape[1], 3), dtype=np.uint8) + 18
        banner_text = f"CottonGuard AI Diagnostic Stack: {cls_name} (Pred: {pred_class}, Conf: {conf*100:.1f}%)"
        cv2.putText(banner, banner_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (34, 197, 94), 2, cv2.LINE_AA)
        full_composite = np.vstack([banner, composite_grid])

        layer5_path = out_dir / f"{c_idx+1:02d}_{stem}_layer5_diagnostic_composite.jpg"
        Image.fromarray(full_composite).save(layer5_path, quality=95)

        results_summary.append({
            "class_name": cls_name,
            "filename": fname,
            "predicted_class": pred_class,
            "confidence": round(float(conf), 4),
            "gpu_latency_ms": round(mean_gpu_lat, 2),
            "cpu_latency_ms": round(mean_cpu_lat, 2),
            "layers": {
                "layer1_original": str(layer1_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "layer2_gradcam": str(layer2_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "layer3_leaf_boundary": str(layer3_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "layer4_lesion_mask": str(layer4_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "layer5_diagnostic_composite": str(layer5_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
        })
        print(f"[{cls_name}] Generated 5-layer stack. GPU Latency: {mean_gpu_lat:.2f} ms | Pred: {pred_class} ({conf*100:.1f}%)")

    metrics_out = PROJECT_ROOT / "artifacts" / "metrics" / "gradcam_evaluation.json"
    with open(metrics_out, "w") as f:
        json.dump({
            "hardware": {
                "gpu_device": str(device_gpu),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
                "mean_gpu_latency_ms": round(float(np.mean(gpu_latencies)), 2),
                "max_gpu_latency_ms": round(float(np.max(gpu_latencies)), 2),
                "mean_cpu_latency_ms": round(float(np.mean(cpu_latencies)), 2),
                "target_latency_budget_ms": 300.0,
                "latency_budget_met": bool(np.mean(gpu_latencies) < 300.0)
            },
            "samples": results_summary
        }, f, indent=2)

    print("\n" + "=" * 50)
    print("EXPLAINABILITY BENCHMARK RESULTS:")
    print(f"  Mean GPU Grad-CAM Latency: {np.mean(gpu_latencies):.2f} ms (Budget: < 300 ms)")
    print(f"  Mean CPU Grad-CAM Latency: {np.mean(cpu_latencies):.2f} ms")
    print(f"  All 7 Classes Evaluated Successfully: {len(results_summary) == 7}")
    print(f"  Visual Showcase Output: {out_dir}")
    print("=" * 50)


if __name__ == "__main__":
    generate_showcase()
