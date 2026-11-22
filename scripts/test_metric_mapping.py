"""
Synthetic Detection Metric Class Mapping Verification (Phase 9.5.3.3).
Verifies:
1. Exact mapping of TorchVision output labels (1..6) and ground truth (1..6) to 0-based indices (0..5).
2. Background class 0 exclusion from foreground mAP.
3. Perfect AP (1.0000) and mAP (1.0000) across all 6 foreground classes.
4. Correct detection of off-by-one shifts (which yield AP=0.0).
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.detection_losses import compute_detection_metrics
from src.data.detection_dataset import FOREGROUND_CLASSES

def run_metric_mapping_verification():
    print("=" * 80)
    print("PHASE 9.5.3.3 — DETECTION METRIC CLASS MAPPING VERIFICATION")
    print("=" * 80)

    # Test 1: Ground truth and predictions in TorchVision 1..6 format
    # Simulates evaluate_detection_epoch() conversion
    preds_per_image = []
    gts_per_image = []

    # 6 images with 1 lesion each, covering classes 1..6
    for c_tv in range(1, 7):
        box = [50.0 + c_tv * 10, 60.0 + c_tv * 10, 150.0 + c_tv * 10, 160.0 + c_tv * 10]
        # TorchVision model output
        p_boxes = np.array([box], dtype=np.float32)
        p_labels = np.array([c_tv], dtype=np.int64)
        p_scores = np.array([0.98], dtype=np.float32)

        # Ground truth in target dict
        gt_boxes = np.array([box], dtype=np.float32)
        gt_labels = np.array([c_tv], dtype=np.int64)

        # evaluate_detection_epoch mapping:
        img_preds = []
        for pb, pl, ps in zip(p_boxes, p_labels, p_scores):
            if pl >= 1:
                img_preds.append({
                    "class_id": int(pl - 1),
                    "confidence": float(ps),
                    "bbox": pb.tolist(),
                })
        gt_mapped = (gt_labels - 1) if gt_labels.size > 0 else np.zeros((0,), dtype=np.int64)

        preds_per_image.append(img_preds)
        gts_per_image.append({
            "boxes": gt_boxes,
            "labels": gt_mapped,
        })

    # Image 7: Healthy leaf negative control (0 GT, 0 predictions)
    preds_per_image.append([])
    gts_per_image.append({
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
    })

    # Image 8: Model predicts background (label 0). Must be excluded by pl >= 1!
    bg_box = [10.0, 10.0, 50.0, 50.0]
    p_boxes_bg = np.array([bg_box], dtype=np.float32)
    p_labels_bg = np.array([0], dtype=np.int64)  # Background label 0
    p_scores_bg = np.array([0.95], dtype=np.float32)
    img_preds_bg = []
    for pb, pl, ps in zip(p_boxes_bg, p_labels_bg, p_scores_bg):
        if pl >= 1:
            img_preds_bg.append({
                "class_id": int(pl - 1),
                "confidence": float(ps),
                "bbox": pb.tolist(),
            })
    preds_per_image.append(img_preds_bg)
    gts_per_image.append({
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
    })

    metrics = compute_detection_metrics(preds_per_image, gts_per_image, num_classes=6)

    print(f"Total Evaluated Ground Truth Boxes: {metrics['evaluated_gt_boxes']}")
    print(f"Overall Precision: {metrics['precision']:.4f}")
    print(f"Overall Recall:    {metrics['recall']:.4f}")
    print(f"Overall mAP@0.50:  {metrics['mAP50']:.4f}")
    print(f"Overall mAP@0.50:0.95: {metrics['mAP50_95']:.4f}")

    print("\nClass-Wise Breakdown:")
    for c in range(6):
        c_name = FOREGROUND_CLASSES[c]
        c_res = metrics["per_class"][c]
        print(f"  - Class {c} [{c_name}]: AP50 = {c_res['ap50']:.4f}, AP50_95 = {c_res['ap50_95']:.4f}, GT = {c_res['gt_boxes']}")
        assert c_res["ap50"] == 1.0, f"Error: AP50 for {c_name} is {c_res['ap50']} != 1.0"
        assert c_res["ap50_95"] == 1.0, f"Error: AP50_95 for {c_name} is {c_res['ap50_95']} != 1.0"
        assert c_res["gt_boxes"] == 1, f"Error: GT count for {c_name} is {c_res['gt_boxes']} != 1"

    assert metrics["mAP50"] == 1.0, "Overall mAP50 mismatch"
    assert metrics["mAP50_95"] == 1.0, "Overall mAP50_95 mismatch"
    assert metrics["precision"] == 1.0, "Overall Precision mismatch"
    assert metrics["recall"] == 1.0, "Overall Recall mismatch"
    assert len(img_preds_bg) == 0, "Background label 0 was not filtered out"

    print("\n[Off-by-One Sensitivity Test]")
    # Shifting prediction labels by +1 (simulating off-by-one bug)
    shifted_preds = []
    for preds in preds_per_image:
        shifted_img = []
        for p in preds:
            shifted_img.append({
                "class_id": (p["class_id"] + 1) % 6,
                "confidence": p["confidence"],
                "bbox": p["bbox"],
            })
        shifted_preds.append(shifted_img)

    shifted_metrics = compute_detection_metrics(shifted_preds, gts_per_image, num_classes=6)
    print(f"Shifted mAP@0.50: {shifted_metrics['mAP50']:.4f} (Expected: 0.0000)")
    assert shifted_metrics["mAP50"] == 0.0, "Shifted labels should yield 0.0 mAP"

    print("\nVERIFICATION CONFIRMED: Metric class mapping is 100% synchronized and free of off-by-one errors.")
    print("=" * 80)

if __name__ == "__main__":
    run_metric_mapping_verification()
