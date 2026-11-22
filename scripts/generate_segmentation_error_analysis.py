"""
Visual Error Analysis for Diseased-Region Segmentation on Development Images.
Generates Ground Truth, Prediction, False Positive Map, False Negative Map, and Overlay
for representative images across each of the 7 disease classes.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(r"D:\Main_Project\cotton_leaf_ai")
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet_resnet34 import UNetResNet34Segmentor, get_segmentation_eval_transforms
from scripts.evaluate_diseased_region_segmentation import post_process_mask

def run_error_analysis():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = PROJECT_ROOT / "artifacts" / "audit_segmentation_improvement" / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_p = PROJECT_ROOT / "artifacts" / "manifests" / "derived_70_image_dataset_manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Select representative samples from each class
    # One from Batch 01/02 and one from Batch 03 where available
    selected_filenames = [
        "BBC00226.jpg", # Bacterial Blight (Batch 01)
        "BBC00001.jpg", # Bacterial Blight (Batch 03)
        "CV00248.jpg",  # Curl Virus (Batch 01)
        "CV00001.jpg",  # Curl Virus (Batch 03)
        "HL00031.jpg",  # Healthy Leaf (Batch 01)
        "HL00138.jpg",  # Healthy Leaf (Batch 03 - outlier FP)
        "HL00257.jpg",  # Healthy Leaf (Batch 03 - perfect 0 FP)
        "HGD00171.jpg", # Herbicide Damage (Batch 01)
        "HGD00002.jpg", # Herbicide Damage (Batch 03)
        "LHJ00185.jpg", # Leaf Hopper Jassids (Batch 01)
        "LHJ00172.jpg", # Leaf Hopper Jassids (Batch 03)
        "LR00309.jpg",  # Leaf Reddening (Batch 01)
        "LR00578.jpg",  # Leaf Reddening (Batch 03)
        "LV00044.jpg",  # Leaf Variegation (Batch 01)
        "LV00003.jpg",  # Leaf Variegation (Batch 03)
    ]

    segmentor = UNetResNet34Segmentor(image_size=512, device=device)
    raw_eval_transform = get_segmentation_eval_transforms(512)
    model = segmentor._ensure_loaded()
    model.eval()

    records_map = {r["filename"]: r for r in manifest["records"]}
    error_summary = []

    for fname in selected_filenames:
        if fname not in records_map:
            continue
        rec = records_map[fname]
        dclass = rec["disease_class"]
        batch = rec["batch"]
        img_path = PROJECT_ROOT / rec["image_path"]
        mask_path = PROJECT_ROOT / rec["mask_path"]

        pil_img = Image.open(img_path).convert("RGB")
        img_rgb = np.array(pil_img)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = gt_mask.shape

        # Model inference at 512x512
        transformed = raw_eval_transform(image=img_rgb)
        tensor = transformed["image"].unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(tensor)
            pred_512 = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        pred_orig = cv2.resize(pred_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        refined_pred = post_process_mask(pred_orig, min_disease_cc_size=50)

        # Extract Class 2 (Diseased Region)
        gt_disease = (gt_mask == 2).astype(np.uint8)
        pred_disease = (refined_pred == 2).astype(np.uint8)

        tp_mask = ((pred_disease == 1) & (gt_disease == 1)).astype(np.uint8)
        fp_mask = ((pred_disease == 1) & (gt_disease == 0)).astype(np.uint8)
        fn_mask = ((pred_disease == 0) & (gt_disease == 1)).astype(np.uint8)

        tp = int(np.sum(tp_mask))
        fp = int(np.sum(fp_mask))
        fn = int(np.sum(fn_mask))
        gt_px = int(np.sum(gt_disease))
        pred_px = int(np.sum(pred_disease))

        intersection = tp
        union = tp + fp + fn
        dice = float(2 * intersection / (2 * intersection + fp + fn)) if (2 * intersection + fp + fn) > 0 else (1.0 if (gt_px == 0 and pred_px == 0) else 0.0)
        iou = float(intersection / union) if union > 0 else (1.0 if (gt_px == 0 and pred_px == 0) else 0.0)

        # Create diagnostic 5-panel visualization:
        # Panel 1: Original RGB
        # Panel 2: GT Mask (Green=Leaf, Red=Disease)
        # Panel 3: Pred Mask (Green=Leaf, Red=Disease)
        # Panel 4: Error Map (Green=TP, Magenta=FP, Cyan=FN)
        # Panel 5: RGB Overlay with Pred Disease in Red and GT boundary in Yellow

        h, w = orig_h, orig_w
        p1_rgb = cv2.resize(img_rgb, (300, 300))

        # Panel 2: GT Mask visualization
        gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
        gt_vis[gt_mask == 1] = [0, 200, 0]    # Green = Healthy leaf
        gt_vis[gt_mask == 2] = [255, 0, 0]    # Red = Disease
        p2_gt = cv2.resize(gt_vis, (300, 300))

        # Panel 3: Pred Mask visualization
        pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
        pred_vis[refined_pred == 1] = [0, 200, 0] # Green = Healthy leaf
        pred_vis[refined_pred == 2] = [255, 0, 0] # Red = Disease
        p3_pred = cv2.resize(pred_vis, (300, 300))

        # Panel 4: Error map (TP=Green, FP=Magenta [255, 0, 255], FN=Cyan [0, 255, 255])
        err_vis = np.zeros((h, w, 3), dtype=np.uint8)
        err_vis[tp_mask == 1] = [0, 255, 0]     # TP = Green
        err_vis[fp_mask == 1] = [255, 0, 255]   # FP = Magenta
        err_vis[fn_mask == 1] = [0, 255, 255]   # FN = Cyan
        p4_err = cv2.resize(err_vis, (300, 300))

        # Panel 5: RGB overlay
        overlay = img_rgb.copy()
        overlay[pred_disease == 1] = (0.5 * overlay[pred_disease == 1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        # Draw GT contour in yellow
        contours, _ = cv2.findContours(gt_disease, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
        p5_over = cv2.resize(overlay, (300, 300))

        # Combine into horizontal strip: (300 x 1500)
        combined = np.hstack([p1_rgb, p2_gt, p3_pred, p4_err, p5_over])

        # Add top banner
        banner = np.zeros((50, 1500, 3), dtype=np.uint8)
        text = f"{fname} [{batch}] - {dclass} | Dice: {dice:.4f} | IoU: {iou:.4f} | GT Disease: {gt_px:,} px | Pred Disease: {pred_px:,} px (TP:{tp:,}, FP:{fp:,}, FN:{fn:,})"
        cv2.putText(banner, text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Sub-labels for panels
        sub_labels = ["1. Original RGB", "2. Ground Truth", "3. Predicted Mask", "4. Error (TP:Grn, FP:Mag, FN:Cyan)", "5. Overlay (Pred:Red, GT:Yel)"]
        for p_idx, lbl in enumerate(sub_labels):
            cv2.putText(combined, lbl, (p_idx * 300 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        final_img = np.vstack([banner, combined])
        out_path = out_dir / f"error_analysis_{fname.split('.')[0]}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

        error_summary.append({
            "filename": fname,
            "batch": batch,
            "disease_class": dclass,
            "gt_disease_pixels": gt_px,
            "pred_disease_pixels": pred_px,
            "tp": tp, "fp": fp, "fn": fn,
            "dice": round(dice, 4),
            "iou": round(iou, 4),
            "saved_image": str(out_path.name)
        })
        print(f"[{batch}] {fname:<12} | {dclass:<24} | Dice: {dice:.4f} | GT: {gt_px:>7,d} | Pred: {pred_px:>7,d} -> {out_path.name}")

    with open(out_dir / "error_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(error_summary, f, indent=2)
    print(f"\n[DONE] Error analysis saved to {out_dir}")

if __name__ == "__main__":
    run_error_analysis()
